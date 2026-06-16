# A-M3 · Doing 任务详细规格

> Topic：显存账本（四块字节数）+ ZeRO Stage 1/2/3 各分什么 + FSDP 在 PyTorch 里的实现 + activation checkpointing 的 compute↔memory 取舍。
>
> 三个任务：`mem_calc.py` = 自己推显存公式 + 实测验证；`train_fsdp.py` = DDP→FSDP-grad→FSDP-full→FSDP+ckpt 渐进升级；`mem_breakdown.md` = 把每一步的显存收益归因到具体哪一块（params/grads/opt-state/activations）。

---

## 任务 1 · `mem_calc.py` 显存账本计算 + 实测验证

### 为什么做这个
"这个模型能不能塞进这张卡？"是开训前每次都要回答的问题，但很多人只能靠跑一把 OOM 来试。其实训练显存是可以掰开算清楚的——参数、梯度、优化器状态、激活值各占多少字节都有公式。把这四块亲手推一遍、再用实测对照验证误差在 10% 以内，你就拥有了不开机就能估显存的能力，这既是常考的面试题，也是后面选多大模型、要不要省显存技巧的判断依据。算的过程里你还会发现激活值常常占了一半以上，这正好引出下一个任务为什么需要 activation checkpointing。

### 目标
能徒手算出"训练 N 参数模型 + AdamW + bf16 大概要多少显存"。这是面试常考题，也是 A-M4 选模型大小时的判断依据。

### Sub-tasks

1. **写 `mem_calc(...)`**——返回四块显存（bytes）。签名 + 返回类型：
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
       batch: int,
       seq_len: int,
       precision: str = "bf16-mixed",   # "bf16-mixed" | "fp32"
   ) -> MemBreakdown:
       """单卡满血四块（未分片）"""
       ...
   ```
   设计要点（职责分离）：
   - **`mem_calc` 只算单卡满血四块**——config → 四块 bytes，6 参数，不管并行。单一职责。
   - **分片是 `MemBreakdown.shard()` 方法**——`MemBreakdown → MemBreakdown` 的变换（满血 → 每卡），放数据类上天经地义；MemBreakdown 自己知道哪几块能切。**不要把 strategy/n_gpus 塞进 mem_calc**（那会让它既算又分、职责混，且对比三策略时要重算 base）。
   - **用法**：`base = mem_calc(...)` 算一次满血 → `base.shard("ddp", 8)` / `base.shard("fsdp_full", 8)` 各分一下，**base 不重算**。A-M3 的对比表（DDP vs FSDP-grad vs FSDP-full）正好是 `base.shard(各策略)`。
   - **分片规则**（`shard()` 内部）：
     - `ddp`：四块都不分（每卡完整）
     - `fsdp_grad`（ZeRO-2）：grads + opt_state ÷ n_gpus；params 不分
     - `fsdp_full`（ZeRO-3）：params + grads + opt_state ÷ n_gpus
     - **activations 任何策略都不分**——按各卡 micro-batch 各算各的，不是被切的对象
   - **6 个参数全必要**：`n_params`（params/grads/opt-state）+ `n_layer/n_embd/batch/seq_len`（activations）+ `precision`（字节数）。不传 `n_head`/`head_dim`——activation 简化公式用 `n_embd` 就够。
   - **存 bytes（int），不存 GB**——bytes 精确、和 `max_memory_allocated()` 同单位，对照时直接比。`as_gb()` 只在展示时转，且**返回每块 + total 的 dict**（不是只返回 total）——这样打印对比表时每块的 GB 直接取，不用在外面手动 `/GB`。
   - **optimizer 写死 AdamW**（`8×n_params`）——LLM 标配，想对比 SGD 再加参数。
   - `total` 是 property（派生，不存）。

   四块各自怎么算（bf16-mixed + AdamW，**分片前的单卡满血值**，再按 strategy ÷ n_gpus）：
   - **Params**（fp32 或 bf16）：`n_params × bytes_per_param`（fp32=4, bf16=2）
     - 注意 mixed precision 下其实 params 有两份：bf16 用于 forward/backward，fp32 master copy 给 optimizer——所以是 `n_params × (2 + 4) = 6 × n_params` 字节
   - **Grads**：**统一按 fp32 master grad 算 = `4 × n_params`，不随 precision 变**（即使 bf16-mixed，optimizer 更新用的也是 fp32 master grad）。代码里 grads 这一项写死 4 字节、不看 precision——这是有意的，注释说明即可。
   - **Optimizer state (AdamW)**：每个参数两个 fp32 buffer（first moment m + second moment v）→ `8 × n_params`
     - 对比：SGD with momentum = `4 × n_params`；AdaFactor 更省（稀疏化）
   - **Activations**（最复杂）：transformer 每层 forward 要存下来给 backward 用的中间结果。**就用这个简化公式**（参数和上面签名一致，不引入 n_head）：
     ```
     activation_bytes ≈ CONST × n_layer × batch × seq_len × n_embd × 2
     ```
     - `× 2`：bf16，每个中间值 2 字节
     - `CONST`：每层要存的中间值个数的经验系数（attention scores / softmax 输出 / FFN 中间 / norm 输入 / residual 等加总）。**这个数没有标准答案**——文献里 ~10 到 ~34 都有，取决于你的实现（dropout mask 存不存、attention 是否 fused、激活中间值留不留）。**所以不要信任何一个固定值**：先随便取个 10–16 算个量级，然后用实测标定——这才是本任务的核心（见 sub-task 5）
     - **标定法**（A-M3 真正要做的）：实测 `max_memory_allocated()` 减去 params+grads+opt_state（这三块公式是确定的），剩下的就是你这套实现的真实 activation；反推出**你自己的** CONST。这个标定出来的数才可信，公式只是给量级
     - 为什么不用 Megatron 论文的完整公式（带 `5 × a × s / h`、`24/t` 这些项）：那个需要 n_head 和 tensor_parallel_size，跟本任务砍掉 n_head 的签名冲突，且精度对教学是过度的。简化版自洽且够用
     - 验算：nanoGPT-size（n_layer=12, n_embd=768, seq=1024, batch=8）代入，`12 × 12 × 8 × 1024 × 768 × 2 ≈ 1.7 GB`——落在合理量级

2. **总显存** = params + grads + opt_state + activations（就是 `MemBreakdown.total`）。
   > `mem_calc` 只算这四块可预测的。实测 `max_memory_allocated()` 通常会比这个 total **高一截**——多出来的是 CUDA context（几百 MB）、cuDNN workspace、NCCL buffers、allocator 碎片，这些不随模型大小线性变、没有干净公式，**不要在 mem_calc 里硬塞一个固定 GB 数**。它们会在你标定 activation CONST 时被一起吸收进去（实测 - 四块 = activation + 杂项，统一用 CONST 拟合），所以不单列。

3. **分片**：见上面 `MemBreakdown.shard(strategy, n_gpus)`——`mem_calc` 算的满血 `base`，调 `base.shard(...)` 得每卡值。三种策略各分一次拼成对比表（task 3 的 `mem_breakdown.md` 就靠这个）。

4. **写 `mem_calc.py` CLI**（参数与 `mem_calc` 签名一致 + 一个 `--strategy/--n_gpus` 给 shard 用）：`python mem_calc.py --n_params 350M --n_layer 24 --n_embd 1024 --batch 8 --seq 1024 --precision bf16-mixed --strategy fsdp_full --n_gpus 8`，内部 `mem_calc(...).shard(strategy, n_gpus)` 拿每卡 `MemBreakdown`，用 `.as_gb()` 打印四块 + total

5. **实测验证 / 标定 CONST（可选——可并入 task 2 一起做，不用单独跑一次 GPU）**：
   - mem_calc 的四块里，params/grads/opt 公式是**确定的**；只有 activation 的 `CONST` 是估的。标定就是用实测反推你这套实现的真实 CONST。
   - **怎么标定**（如果做）：**用 DDP 或单卡跑**（不要用 FSDP——分片 + all-gather 临时 buffer 会污染实测，反推不干净）。在 train loop 跑稳后：
     ```python
     torch.cuda.reset_peak_memory_stats()
     # 跑 10 step
     peak = torch.cuda.max_memory_allocated()    # bytes，跟 mem_calc 同单位
     ```
   - `实测 peak − (params + grads + opt)[公式确定] = 真实 activation + 杂项` → 反推 CONST。
   - **但这步可选**：A-M3 的知识点（四块账本 + FSDP 分片递进）你做完 mem_calc.py 就拿到了；CONST 的具体值（16 还是 20）只对你这套实现有效、是细节。**按"不死磕"原则，可跳过单独标定**。
   - **更省的做法**：task 2 反正要真跑 DDP→FSDP 渐进、记每种 strategy 的实测 peak——那时顺手跟 mem_calc 预测对一下**量级**（不追 <10%，看数量级对不对）即可，不用在这里单独跑。

6. **进阶（选做）**：用 `torch.cuda.memory._record_memory_history(enabled='all')` + `_dump_snapshot('out.pickle')`，上传到 https://pytorch.org/memory_viz 看每一笔 alloc/free 的来源——能看到 activation 的具体大头是哪一层的什么 op

### 成功标准
- `mem_calc.py` 给出分片后每卡的四块 + total（`mem_calc(...).shard(strategy, n_gpus).as_gb()`），CLI 接 `--n_params --n_layer --n_embd --batch --seq --precision`（mem_calc 参数）+ `--strategy --n_gpus`（shard 参数）
- 跑出 ddp / fsdp_grad / fsdp_full 三种 strategy 的每卡显存对比表，能看出 ZeRO-2→ZeRO-3 grads/opt 先分、params 后分的递进
- 能说出："activation 占了总显存多少"——350M 模型 activation 常占一半以上，这是下一步要 activation checkpointing 的动机
- （可选）实测 vs 理论量级对照——见 sub-task 5，可并入 task 2

### 失败排查
- **理论比实测高很多**：你忘了 grad 实际只在 backward 那一刻峰值，optimizer state 实际只在 step 那一刻峰值——peak memory 是这些瞬时峰的最大，不是和
- **理论比实测低很多**：activation 公式 constant 取小了；或者你忘记 dropout / residual / layernorm 各自都要存中间值
- **跑出来 OOM 但 mem_calc 说应该够**：cuDNN workspace / NCCL buffer 没算（再加 1-2 GB 经验值）；或者用了 `torch.compile` 编译时内存峰值更高

### 辅助阅读（非 canonical）
- HuggingFace "Model anatomy"（详细推 activation 公式）：https://huggingface.co/docs/transformers/model_memory_anatomy
- Megatron-LM activation 公式原文（论文 §4）：https://arxiv.org/abs/2205.05198
- PyTorch memory viz：https://pytorch.org/memory_viz

### Deliverable
- `course-a/m3-fsdp/mem_calc.py` + `course-a/m3-fsdp/mem_calc_validation.md`（理论 vs 实测表格 + 一段解释）

---

## 任务 2 · DDP → FSDP 渐进升级 + activation checkpointing

### 为什么做这个
DDP 的问题是每张卡都存一整份参数、梯度、优化器状态，模型一大就单卡 OOM。FSDP（也就是 ZeRO）的思路是把这些东西切片分到各张卡上，谁用到再临时聚回来——这是今天训练超大模型的主流做法。这里不是直接上最激进的配置，而是从 DDP 一步步升到"切梯度"、"切参数"、再叠加 activation checkpointing，每升一级都实测显存和吞吐怎么变。这样你不只是会调 API，而是能把每一步省下的显存对应到上个任务算的具体哪一块，真正理解 ZeRO 各个 stage 到底在切什么、代价是什么。

### 目标
把 mygpt 升到 350M（或 1B 看资源），实测 4 种配置的 (peak memory, tokens/sec/GPU)：
1. DDP（baseline）
2. FSDP `SHARD_GRAD_OP`（ZeRO-2）
3. FSDP `FULL_SHARD`（ZeRO-3）
4. FSDP `FULL_SHARD` + activation checkpointing

### Sub-tasks

1. **模型升级到 350M**：mygpt config 改 `n_layer=24, n_head=16, n_embd=1024, block_size=1024`（≈ 350M）；如果资源够 1B 就 `n_layer=24, n_head=16, n_embd=2048`

2. **配置 1 — DDP baseline**：跑 A-M2 的 train_ddp.py 在 350M 上。**这一步可能 OOM**——OOM 就把 micro-batch 调到能跑（4 / 2 / 1），记录最大可行 micro-batch 和对应 peak memory

3. **配置 2 — FSDP SHARD_GRAD_OP**（ZeRO-2，不切 params）：
   ```python
   from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
   from torch.distributed.fsdp import ShardingStrategy
   # FSDP2 推荐写法：
   for layer in model.transformer.h:
       fully_shard(layer, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
   fully_shard(model)
   ```
   或 FSDP1（材料里更多）：
   ```python
   from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
   from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
   model = FSDP(
       model,
       sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
       auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
       mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
   )
   ```
   **重要**：`auto_wrap_policy=transformer_auto_wrap_policy` 必须用，否则整个模型当一个 shard 单元，通信粒度极差

4. **配置 3 — FSDP FULL_SHARD**（ZeRO-3）：把 sharding_strategy 改成 `FULL_SHARD`

5. **配置 4 — FSDP FULL_SHARD + activation checkpointing**：
   ```python
   from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, CheckpointWrapper
   apply_activation_checkpointing(model, check_fn=lambda m: isinstance(m, Block))
   ```
   这会让每个 transformer block 在 backward 时重算 forward（用 compute 换 memory）

6. **每个配置跑 ≥ 200 step**，记 (peak memory GB, tokens/sec/GPU, val loss after N step)

### 成功标准
- 4 组数字给全的表格
- 显存递减：DDP > FSDP-grad > FSDP-full > FSDP-full+ckpt
- 4 跟 3 对比：peak memory 应该降 30-50%，throughput 降 20-30%（activation 重算的代价）
- 解释："为什么 FSDP-grad 比 DDP 显存少了 X GB"——对应 mem_calc 里的 grads + opt_state 切了 1/N

### 失败排查
- **FSDP 启动报 `auto_wrap_policy` 错**：FSDP1 和 FSDP2 API 不同；FSDP1 要传 `transformer_layer_cls={YourBlockClass}` 集合；FSDP2 要逐层 `fully_shard()`。混用会出奇怪错误
- **FSDP 比 DDP 慢得过分**：未配 `auto_wrap_policy` → 退化成全模型一个 shard、通信巨慢
- **activation checkpointing 显存没怎么降**：check_fn 写错了（没匹配到任何 block）；或者你用了 `reentrant=True`（PyTorch 默认，但 FSDP 推荐 `use_reentrant=False`）

### 辅助阅读（非 canonical）
- PyTorch FSDP2 docs（新 API）：https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- `apply_activation_checkpointing` 用法：https://pytorch.org/docs/stable/checkpoint.html
- "How to choose between FSDP1 and FSDP2"：搜博客即可

### Deliverable
- `course-a/m3-fsdp/train_fsdp.py`（接 `--strategy {ddp, fsdp_grad, fsdp_full, fsdp_full_ckpt}` flag 切换）
- `course-a/m3-fsdp/strategy_comparison.md`：4 组数字表 + 一段对每相邻两步的差异解释

---

## 任务 3 · 显存收益归因表

### 为什么做这个
任务 2 你看到了"FSDP-full 比 DDP 省了一大截显存"，但如果只停在"省了 X GB"这个结论上，面试时被追问"具体省在哪"就会卡壳。这个任务逼你把每个配置的四块显存（参数/梯度/优化器状态/激活值）逐项列成表，让"为什么这一步降幅最大"变成一眼能看懂的账本。做完你会清楚地看到：切参数是降显存最狠的一步，而激活值不会因为切片自动变小、必须靠 checkpointing——这种把现象拆到具体成因的能力，正是区分"会用工具"和"真懂原理"的地方。

### 目标
用任务 1 的 `mem_calc` + 任务 2 的实测，把 4 个配置每一块（params / grads / opt-state / activations）的字节数列出来，让"为什么 FSDP-full 比 DDP 省 X GB"变成可读的表。

### Sub-tasks
1. 对 4 个配置（DDP / FSDP-grad / FSDP-full / FSDP-full+ckpt），各自填表：

   | 配置 | params | grads | opt_state | activations | 总 |
   |---|---|---|---|---|---|
   | DDP | 6×N | 4×N | 8×N | A | 18N + A |
   | FSDP-grad | 6×N | 4×N/G | 8×N/G | A | 6N + 12N/G + A |
   | FSDP-full | 6×N/G | 4×N/G | 8×N/G | A | 18N/G + A |
   | FSDP-full+ckpt | 6×N/G | 4×N/G | 8×N/G | A/k | 18N/G + A/k |

   N = n_params, G = n_gpus, A = activation_bytes, k = checkpoint 折扣（一般 4-8）
2. 用 350M / 8 GPU / bf16 mixed 算具体数字，与任务 2 的实测对照
3. README 一段话：解释为什么 FSDP-full 把"params 分掉"是显存降幅最大的一步

### 成功标准
- 表格四行四列填全
- 理论值 vs 实测差异 < 15%
- 一段解释提到："activation 不会因为 FSDP 自动降低，必须靠 checkpointing"

### Deliverable
- `course-a/m3-fsdp/mem_breakdown.md`（这个表 + 解释）

---

## 三个任务做完之后

- 跑 QUIZ.md
- A-M3 过关 → 开 A-M4（在 AWS 多 GPU 上把模型扩到 1B）
