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

1. **写 `mem_calc(n_params, optimizer, precision, batch, seq_len, n_layer, n_head, head_dim) -> dict`**——返回四块字节数：
   - **Params**（fp32 或 bf16）：`n_params × bytes_per_param`（fp32=4, bf16=2）
     - 注意 mixed precision 下其实 params 有两份：bf16 用于 forward/backward，fp32 master copy 给 optimizer——所以是 `n_params × (2 + 4) = 6 × n_params` 字节
   - **Grads**：和 params 同精度（mixed precision 下 grad 是 bf16，但 optimizer 用的 master grad 是 fp32 → 也是 `4 × n_params`）
   - **Optimizer state (AdamW)**：每个参数两个 fp32 buffer（first moment m + second moment v）→ `8 × n_params`
     - 对比：SGD with momentum = `4 × n_params`；AdaFactor 更省（稀疏化）
   - **Activations**（最复杂）：transformer 每层 forward 保存的中间结果。粗略估算：
     - 每层活动 ≈ `s × b × h × (10 + 24/t + 5 × a × s / h)` bytes（Megatron 论文公式简化版，s=seq_len, b=batch, h=n_embd, a=n_head, t=tensor_parallel_size——这里 t=1）
     - 简化版（fp16 mixed precision、no recompute）：`activation_bytes ≈ 2 × s × b × h × n_layer × constant`，constant 取 ~16-34 看实现
     - 推荐做法：用 nanoGPT-size 配置（n_layer=12, n_head=12, n_embd=768, seq=1024, batch=8）算个具体值，目标 ~1-3 GB

2. **总显存** = params + grads + opt_state + activations + 杂项（NCCL buffers、cuDNN workspace 等，估 ~1-2 GB）

3. **DDP / FSDP 修正**：
   - DDP：每张卡都有完整 params/grads/opt-state；只 activation 因为 micro-batch 切了
   - FSDP `SHARD_GRAD_OP` (ZeRO-2)：grads + opt_state 切成 1/N，params 不切
   - FSDP `FULL_SHARD` (ZeRO-3)：params + grads + opt_state 都切成 1/N

4. **写 `mem_calc.py` CLI**：`python mem_calc.py --n_params 350M --opt adamw --precision bf16-mixed --batch 8 --seq 1024 --n_layer 24 ... --strategy fsdp_full --n_gpus 8`，打印四块字节数 + 总和

5. **实测验证**：
   - 跑你的 mygpt（350M 配置），在 train loop 第 100 step 末尾调用：
     ```python
     torch.cuda.reset_peak_memory_stats()
     # 跑 10 step
     peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
     ```
   - 用 `mem_calc` 算一份理论值
   - 实测 vs 理论：差异 < 10% 算过关；差太多说明你少算了某块（最常见是 activation 公式的 constant 不对）

6. **进阶（选做）**：用 `torch.cuda.memory._record_memory_history(enabled='all')` + `_dump_snapshot('out.pickle')`，上传到 https://pytorch.org/memory_viz 看每一笔 alloc/free 的来源——能看到 activation 的具体大头是哪一层的什么 op

### 成功标准
- `mem_calc.py` 给出四块 + 总和（GB），CLI 接 `--n_params --opt --precision --batch --seq --strategy --n_gpus` 等参数
- 一组实测 vs 理论的对比表，差异 < 10%
- README 一段话："activation 占了总显存的多少"——会发现 350M 模型的 activation 经常占一半以上，这是 A-M3 选 activation checkpointing 的动机

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
