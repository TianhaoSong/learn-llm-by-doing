# A-M3 · Doing 任务详细规格

> Topic：显存账本（四块字节数）+ ZeRO Stage 1/2/3 各分什么 + FSDP 在 PyTorch 里的实现 + activation checkpointing 的 compute↔memory 取舍。
>
> 两个任务：`mem_calc.py` = 自己推显存账本（四块字节数 + 分片）；`fsdp_train.py` sweep = 真跑 DDP→zero2→zero3→±ckpt，实测 peak 对照预测、把显存收益归因到具体哪一块。

---

## 任务 1 · `mem_calc.py` 显存账本计算

### 为什么做这个
"这个模型能不能塞进这张卡？"是开训前每次都要回答的问题，但很多人只能靠跑一把 OOM 来试。其实训练显存能掰开算清楚——参数、梯度、优化器状态、激活值各占多少字节都有公式。把这四块亲手推一遍，你就有了不开机就能估显存、并据此选模型大小/并行策略的能力，这也是常考的面试题。算的过程里你会发现激活值常占大头，这正好引出下一个任务为什么需要 activation checkpointing。

### 目标
能徒手算出"训练 N 参数模型 + AdamW + 给定精度 大概要多少显存"，并能把它按并行策略分片到每卡。

### Sub-tasks

1. **写 `mem_calc(...)` + `MemBreakdown`**——返回四块显存（bytes）：
   ```python
   from dataclasses import dataclass, asdict

   @dataclass
   class MemBreakdown:
       """训练显存四大块，单位 bytes。"""
       params: int
       grads: int
       opt_state: int
       activations: int

       @property
       def total(self) -> int:
           return self.params + self.grads + self.opt_state + self.activations

       def shard(self, strategy: str = "ddp", n_gpus: int = 1) -> "MemBreakdown":
           """按并行策略 ÷ n_gpus，返回每卡的 breakdown。activations 不分。"""
           p, g, o = self.params, self.grads, self.opt_state
           if strategy == "fsdp_grad":      # ZeRO-2: 分 grads + opt_state
               g //= n_gpus; o //= n_gpus
           elif strategy == "fsdp_full":    # ZeRO-3: 再分 params
               p //= n_gpus; g //= n_gpus; o //= n_gpus
           # ddp: 四块都不分
           return MemBreakdown(p, g, o, self.activations)

       def as_gb(self) -> dict[str, float]:
           """展示用：各块（含 total）转 GB。"""
           gb = {k: v / 1024**3 for k, v in asdict(self).items()}
           gb["total"] = self.total / 1024**3
           return gb


   def mem_calc(
       n_params: int,
       n_layer: int,
       n_embd: int,
       n_head: int,
       batch: int,
       seq_len: int,
       precision: str = "bf16-mixed",   # "bf16-mixed" | "fp32"
   ) -> MemBreakdown:
       """单卡满血四块（未分片）"""
       ...
   ```

   设计要点（职责分离）：
   - **`mem_calc` 只算单卡满血四块**——config → 四块 bytes，不管并行。单一职责。
   - **分片是 `MemBreakdown.shard()` 方法**——`MemBreakdown → MemBreakdown` 的变换（满血 → 每卡），放数据类上：MemBreakdown 自己知道哪几块能切。别把 strategy/n_gpus 塞进 mem_calc（会让它既算又分、职责混，且对比三策略时要重算 base）。
   - **用法**：`base = mem_calc(...)` 算一次满血 → `base.shard("ddp", G)` / `base.shard("fsdp_full", G)` 各分一下，base 不重算。对比表（DDP vs ZeRO-2 vs ZeRO-3）正好是 `base.shard(各策略)`。
   - **存 bytes（int），不存 GB**——bytes 精确、和 `torch.cuda.max_memory_allocated()` 同单位，对照时直接比；`as_gb()` 只在展示时转，返回每块 + total 的 dict（这样打印对比表每块的 GB 直接取，不用在外面手动 `/GB`）。
   - **optimizer 写死 AdamW**（`8×n_params`）——LLM 标配，想对比 SGD 再加参数。
   - `total` 是 property（派生，不存）。
   - **参数为什么是这几个**：`n_params` 定 params/grads/opt-state；`n_layer/n_embd/batch/seq_len/n_head` 定 activations；`precision` 定字节数。`head_dim` 不传（= n_embd/n_head，冗余）。

2. **四块各自怎么算**（bf16-mixed + AdamW，**分片前的单卡满血值**）：

   > **记号约定**：`6×N` / `4×N` / `8×N` 里 **N = n_params（参数量）**，前面的 **6/4/8 = 每个参数占的字节数**，乘出来单位是 **bytes**。例：350M 参数的 opt_state = `8 bytes/param × 350M = 2.8 GB`。

   **model state（params + grads + opt_state）——只跟参数量有关**：

   | 块 | 字节/参数 | = bytes | 为什么 |
   |---|---|---|---|
   | **Params** | 6 | `6 × N` | bf16-mixed 有两份：bf16 算用（2B）+ fp32 master copy 给 optimizer（4B）= 6B；纯 fp32 则是 4B |
   | **Grads** | 4 | `4 × N` | 统一按 fp32 master grad = 4B，**不随 precision 变**（optimizer 更新用 fp32 master grad）。代码里写死 4、不看 precision，注释说明 |
   | **Opt state** | 8 | `8 × N` | AdamW 每参数两个 fp32 buffer（m + v）= 8B。对比：SGD+momentum = 4B、AdaFactor 更省 |

   → 三块合计 **model state = 18 × N bytes**（bf16-mixed + AdamW）。1B 参数就是 18 GB，单卡装不下——这是 ZeRO 的动机。

   **activations——跟 batch/seq 有关（不只跟参数量），必须分两项**：
   ```
   activation_bytes ≈ byte × n_layer × [ C1 × batch × seq_len × n_embd       # 线性项：FFN/norm/residual 等
                                       + C2 × batch × n_head × seq_len² ]      # O(T²) 项：attention 分数矩阵 [B,nh,T,T]
   ```
   - `byte`：fp32=4，bf16=2
   - **为什么必须带 O(T²) 项**：朴素 attention 的 `q @ kᵀ` 会 materialize 整个 `[B, n_head, T, T]` 矩阵。线性项 ∝ seq_len，T² 项 ∝ seq_len²——**当 seq_len 大时（如 1024+），T² 项是 activation 的大头**，漏掉它会把 activation 严重低估。这也是为什么 activation 公式（和 mem_calc 签名）需要 `n_head`。
   - **activation 跟 model state 的区别**（关键概念）：model state 只由参数量定（固定模型就是常数）；activation 还随 **batch / seq_len** 变——同一个模型 batch 翻倍 activation 就翻倍。所以"模型多大"和"一次 forward 要多少 activation"是两回事。
   - `C1` / `C2`：经验系数，没有标准值（取决于 dropout mask 存不存、attention 是否 fused）。**FlashAttention 不 materialize T² 矩阵 → C2≈0**（FlashAttention 是 B 课程内容，这里知道它能砍掉 T² 项即可）。朴素实现 C2 显著。先取量级（C1~10、C2~2），再用实测标定（见 sub-task 4）。

3. **分片**：调 `base.shard(strategy, n_gpus)` 得每卡值。三种策略各分一次拼成对比表：
   - `ddp`：四块都不分（每卡完整 model state + activation）
   - `fsdp_grad`（ZeRO-2）：grads + opt_state ÷ n_gpus，params 不分
   - `fsdp_full`（ZeRO-3）：params + grads + opt_state ÷ n_gpus
   - **activations 任何策略都不分**——它按各卡的 micro-batch 各算各的，不是被切的对象

4. **CLI**：`python mem_calc.py --n-params 350000000 --n-layer 24 --n-embd 1024 --n-head 16 --batch 8 --seq-len 1024 --precision bf16-mixed --strategy fsdp_full --n-gpus 8`，内部 `mem_calc(...).shard(strategy, n_gpus)` 拿每卡 `MemBreakdown`，`.as_gb()` 打印四块 + total。

5. **（可选）用实测标定 activation 系数**：mem_calc 里 model state 三块公式是确定的；只有 activation 的 C1/C2 是估的。标定 = 用实测反推真实系数。
   - **方法**：用 DDP 或单卡跑（别用 FSDP——分片 + all-gather 临时 buffer 会污染实测）。train loop 跑稳后：
     ```python
     torch.cuda.reset_peak_memory_stats(device)   # 训练循环开始前清零，只测稳态峰值
     # 跑若干 step
     peak = torch.cuda.max_memory_allocated(device)   # bytes，跟 mem_calc 同单位
     ```
   - `实测 peak − model state（公式确定） = 真实 activation + 杂项` → 反推 C1/C2。
   - **可跳过**：知识点（四块账本 + 分片）做完 mem_calc 就拿到了，系数具体值只对你这套实现有效。按"不死磕"原则，量级对就行，标定可并入任务 2（任务 2 反正要实测各 strategy 的 peak）。

### 成功标准
- `mem_calc(...).shard(strategy, n_gpus).as_gb()` 能给出 ddp / fsdp_grad / fsdp_full 三种策略的每卡四块 + total
- 看得出 ZeRO-2→ZeRO-3 的递进：先分 grads/opt，再分 params
- 能说清：model state（∝参数量）vs activation（∝batch·seq，含 O(T²) 项）的区别，以及为什么 activation 常占大头

### 失败排查
- **预测比实测高很多**：grad 的峰值只在 backward 那一刻、opt state 只在 step 那一刻——peak 是这些瞬时峰的最大值，不是简单相加
- **预测比实测低很多**：多半是 activation 漏了 O(T²) 项（只算了线性项），或 C1/C2 取小了
- **OOM 但预测说够**：CUDA context / cuDNN workspace / NCCL buffer / allocator 碎片没算进去——这些不随模型大小线性变、没有干净公式，不要硬塞固定 GB；实测时它们会体现在"实测比预测高的那一截"里

### 辅助阅读（非 canonical）
- HuggingFace "Model anatomy"（详细推 activation 公式）：https://huggingface.co/docs/transformers/model_memory_anatomy
- Megatron-LM activation 公式原文（论文 §4）：https://arxiv.org/abs/2205.05198
- PyTorch memory viz：https://pytorch.org/memory_viz

### Deliverable
- `mem_calc.py`（`MemBreakdown` + `mem_calc` + CLI）

---

## 任务 2 · DDP → FSDP 渐进升级 + activation checkpointing

### 为什么做这个
DDP 每张卡都存一整份参数、梯度、优化器状态，模型一大就单卡 OOM。FSDP（即 ZeRO）把这些切片分到各卡、谁用到再临时聚回来——这是今天训练大模型的主流做法。这里不直接上最激进的配置，而是从 DDP 一步步升到"切梯度"、"切参数"、再叠加 activation checkpointing，每升一级实测显存和吞吐。这样你能把每一步省下的显存对应到任务 1 算的具体哪一块，真正理解 ZeRO 各 stage 切什么、代价是什么。

### 目标
把模型升到 ~350M，实测 6 种配置 `{ddp, zero2, zero3} × {±activation_ckpt}` 的 (peak memory, tokens/sec/GPU)，归因到四块。

### Sub-tasks

1. **模型升到 ~350M**：config 改 `n_layer=24, n_head=16, n_embd=1024, block_size=1024`（≈350M）。A-M3 用 350M 足够展示 ZeRO + activation ckpt；更大模型（1B+）留 A-M4。

   > **⚠️ batch_size 必须按显存选，否则各档全 OOM、看不到递减**。朴素 attention（`q@kᵀ` 显式 materialize T×T 矩阵）在 seq=1024 时 activation 极吃显存，**activation 才是瓶颈、不是 model state**。各档每卡显存随 batch 增长，而 activation 这块 ZeRO 切不动——所以 batch 越大，越多档位 OOM。
   >
   > **选 batch 的原则**：让显存最高的 DDP 档也能跑通，这样四档都出数、显存阶梯（DDP > ZeRO-2 > ZeRO-3 > +ckpt）才完整可见。
   > - **单卡 32GB 级（如 V100）**：batch 取小（如 2~4），否则 DDP/ZeRO 档会 OOM、只剩 +ckpt 能跑、对比就废了
   > - **单卡 40/80GB 级（如 A100）**：能用更大 batch
   >
   > 先用 mem_calc 预测各档每卡显存、挑一个让 DDP 也 ≤ 显存×0.8 的 batch，再跑。

2. **配置 — DDP baseline**：跑 A-M2 的 train_ddp.py。记 peak memory。DDP 是显存最高的档，它跑通的 batch 后面几档一定也跑得通。

   > **dtype 跟硬件走**：`AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16`。**A100/H100 才有 bf16 tensor core；V100/T4 一类要用 fp16**。所有 `param_dtype` 用这个变量，别写死 bf16。

3. **配置 — ZeRO-2（SHARD_GRAD_OP，不切 params）**，FSDP2 写法：
   ```python
   from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
   for layer in model.blocks:                       # model.blocks = 你的 transformer 层列表
       fully_shard(layer, mp_policy=MixedPrecisionPolicy(param_dtype=AMP_DTYPE))
   fully_shard(model)
   ```
   或 FSDP1（老 API，材料里更多）：
   ```python
   import functools
   from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
   from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
   model = FSDP(
       model,
       sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
       auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
       mixed_precision=MixedPrecision(param_dtype=AMP_DTYPE, reduce_dtype=AMP_DTYPE, buffer_dtype=AMP_DTYPE),
   )
   ```
   **关键**：按「一个 transformer block」为粒度 wrap（`transformer_layer_cls` 填你自己的 block 类）。否则整个模型当一个 shard 单元，通信粒度极差。

4. **配置 — ZeRO-3（FULL_SHARD）**：sharding_strategy 改 `FULL_SHARD`（FSDP2 里是 `reshard_after_forward=True`）。

5. **配置 — + activation checkpointing**（可叠加在任意 ZeRO stage 上）：
   ```python
   from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
       checkpoint_wrapper, apply_activation_checkpointing,
   )
   apply_activation_checkpointing(
       model,
       checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, preserve_rng_state=False),
       check_fn=lambda m: isinstance(m, Block),     # Block = 你的 transformer block 类
   )
   ```
   每个 transformer block 在 backward 时重算 forward（用 compute 换 activation memory）。

6. **每个配置跑若干 step**，记 (peak memory GB, tokens/sec/GPU)。

### 成功标准
- **6 配置 `{ddp, zero2, zero3} × {±ckpt}` 对比表**：每行 = 预测四块（`mem_calc(...).shard()`）+ 实测 peak_mem + throughput
- 显存递减可见：DDP > ZeRO-2 > ZeRO-3，且每个 +ckpt 严格更低
- **归因到具体块**（核心）：
  - "切参数降显存最狠"——ZeRO-3 把 params 也分掉，对应 mem_calc 的 `6N → 6N/G`；而 ZeRO-2→ZeRO-3 降幅小，因为 grads+opt（18 里的 12）ZeRO-2 已切，只剩 params 6 没切
  - "activation 不会因为 ZeRO 自动降低，必须靠 checkpointing"——+ckpt 在三种 stage 下砍掉的显存量级**相同**，因为 ZeRO 切 model state、ckpt 砍 activation，两个操作作用在不同显存块、互不干扰（正交）
  - ckpt 代价：throughput 降（backward 多一次 forward 重算）
- 实测 vs 预测：方向一致即可；绝对值偏差能解释（activation 的 O(T²) 项、初始化/碎片残留）

### 失败排查
- **FSDP 报 `auto_wrap_policy` 错**：FSDP1/FSDP2 API 不同——FSDP1 传 `transformer_layer_cls={你的Block类}`，FSDP2 逐层 `fully_shard()`，别混用
- **FSDP 比 DDP 慢得过分**：没按 transformer block 粒度 wrap → 退化成全模型一个 shard、通信巨慢
- **activation checkpointing 显存没降**：check_fn 没匹配到任何 block；或用了 `use_reentrant=True`（推荐 `False`）
- **各档全 OOM**：batch 太大——见 sub-task 1，activation 是瓶颈，单卡 32GB 级要用小 batch

### 辅助阅读（非 canonical）
- PyTorch FSDP2 docs（新 API）：https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- `apply_activation_checkpointing` 用法：https://pytorch.org/docs/stable/checkpoint.html

### Deliverable
- `fsdp_train.py`（`--strategy {ddp, zero2, zero3}` + `--activation_ckpt`）+ sweep 脚本
- `fsdp_sweep_results.md`：6 配置的（预测四块 + 实测 peak + throughput）表 + 归因分析

---

## 两个任务做完之后

- 跑 QUIZ.md
- A-M3 过关 → 开 A-M4（在 AWS 多 GPU 上把模型扩到更大规模）
