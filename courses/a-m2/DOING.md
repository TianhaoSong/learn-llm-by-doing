# A-M2 · Doing 任务详细规格

> Topic：从单卡跨到多卡。NCCL collective ops 实测、DDP 内部机制（bucket / overlap）、grad accumulation × micro-batch、mixed precision 数值与显存收益。
>
> 五个任务：`bench_collectives.py` = 通信原语带宽实测；`train_ddp.py` = 把 mygpt 改成 DDP；grad accumulation 实验 = 学 `no_sync` 用法；mixed precision 实验 = fp32 vs bf16 vs fp16 三方对比；profiler trace = 学看 timeline 找 overlap。

---

## 任务 1 · `bench_collectives.py` — NCCL 原语带宽实测

### 为什么做这个
多卡训练慢下来时，很大一块时间花在 GPU 之间互相传梯度上，但这个"传一次要多久"对大多数人是个黑盒。亲手测一下不同大小的数据在卡间同步要多少时间、能跑出多少带宽，你就建立起了"一次梯度同步大概几毫秒"的直觉。这个直觉是后面理解 DDP/FSDP 为什么这么设计的前提——比如为什么要把很多小梯度攒成一大块再发，答案就藏在你测出的"小数据带宽极低"这条数据里。

### 目标
亲手测 all-reduce / reduce-scatter / all-gather 在不同 tensor size 下的带宽，建立"DDP 一次 all-reduce 大概多少时间"的直觉。

### Sub-tasks
1. 起 `torch.distributed` 进程组（`init_process_group(backend='nccl')`），从 `torchrun --nproc_per_node=N` 启动
2. 对每个 op（all-reduce / reduce-scatter / all-gather）：
   - tensor size 扫 [1MB, 4MB, 16MB, 64MB, 256MB, 1GB]（fp32 → 元素数 = bytes / 4）
   - 每个 size warmup 5 次后跑 20 次取 median，用 `torch.cuda.Event` 计时
   - 算"算法带宽"：`bytes_moved / time`——注意每个 op 的 `bytes_moved` 公式不同（all-reduce 是 `2 × (N-1)/N × tensor_size`）
3. 打印表格：`(op, tensor_size, time_ms, algo_bw_GBps, bus_bw_GBps)`
4. **记录硬件 + 互联类型**（bench 数字必须标，否则不可比）。先 `nvidia-smi topo -m` 看卡间连接：
   - `PIX`/`PHB`/`NODE` = **PCIe**（g4dn T4、g5 A10G）
   - `NV#` = **NVLink**（p3 V100、p4d A100）
   - 带宽数量级（绝对值看卡，但**互联类型决定档位**）：

   | 互联 | 实例例子 | all-reduce 1GB 算法带宽量级 |
   |---|---|---|
   | PCIe Gen3/4 | g4dn(T4)、g5(A10G) | ~5–15 GB/s |
   | NVLink | p3(V100) | ~50–120 GB/s |
   | NVSwitch | p4d(A100) | ~150–250 GB/s |

### 成功标准
- 跑出表格 + **标注你用的实例和互联类型**（PCIe / NVLink），数字落在上表对应档位即可——不追绝对值
- **核心结论**：表格里能看出 small tensor 是 latency-bound（带宽低）、large tensor 趋近峰值
- 如果你只跑了一种实例（如 g4dn 4×T4 PCIe），那就只填 PCIe 行——**不要求跑遍所有互联类型**；NVLink 档位作为对照知道量级即可（面试讲"我跑的是 PCIe，NVLink 会快一个数量级"）

### 失败排查
- **小 tensor 带宽极低（< 1 GB/s）**：正常——latency-bound；这恰好是为什么 DDP 要用 bucket（小梯度合并成大 tensor 一起 all-reduce）
- **NCCL hang 不动**：`NCCL_DEBUG=INFO` 看 transport；常见原因是 `CUDA_VISIBLE_DEVICES` 或 placement group 不对，导致一些 GPU 看不到对方
- **每次跑数字差很大**：warmup 不够 / 别的进程占着 GPU；每 size warmup 提到 10 次

### 辅助阅读（非 canonical）
- NVIDIA `nccl-tests` README（公认带宽 bench 工具，看它怎么算 bus bandwidth）：https://github.com/NVIDIA/nccl-tests
- NCCL FAQ "What's the difference between algorithm bandwidth and bus bandwidth?": https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/performance.html

### Deliverable
- `course-a/m2-ddp/bench_collectives.py`
- `course-a/m2-ddp/bench_collectives_results.md`：硬件信息 + 三个 op × 6 个 size 的表格 + 一段观察（小 tensor / 大 tensor 行为差异）

---

## 任务 2 · `train_ddp.py` — DDP 版 mygpt + scaling 曲线

### 为什么做这个
"加几张卡训练就快几倍"听起来理所当然，但现实里几乎从来达不到完美的线性加速——卡越多，花在同步上的时间占比越大，多出来的卡有一部分被通信吃掉了。把单卡的 mygpt 改成多卡 DDP、跑出 2/4/8 卡的实际吞吐曲线，你就能亲眼看到"加倍卡数到底换来多少加速"，并理解差距是怎么来的。能把 scaling efficiency 算出来、解释清楚为什么不到 100%，是面试里聊分布式训练绕不开的基本功。

### 目标
把 A-M1 的 mygpt 改成 DDP 版，跑 2 / 4 / 8 GPU，给出 scaling efficiency 曲线。理解 DDP 内部 bucket + backward overlap。

### Sub-tasks
1. **改造 train.py → train_ddp.py**：
   - `init_process_group(backend='nccl')` + `local_rank = int(os.environ['LOCAL_RANK'])` + `torch.cuda.set_device(local_rank)`
   - `model = DDP(model, device_ids=[local_rank])`
   - 只在 `rank == 0` 打 log / 存 checkpoint
2. **scaling 实验**：固定 micro-batch（如 32），跑 N=1/2/4/8 GPU，每个 N 跑 200 step 取 median tokens/sec
3. **算 scaling efficiency**：
   - `efficiency(N) = total_tokens_per_sec(N) / (total_tokens_per_sec(1) × N)`
   - 即「N 卡实际总吞吐」÷「单卡吞吐 × N（线性理想值）」
   - efficiency = 1 是完美线性扩展；< 1 的差距来自通信开销（all-reduce 没完全和 backward overlap），N 越大通常越低
   - 打成表即可（N, total_tokens/sec, efficiency），画不画图随意
4. **bucket size 实验**（选做）：`DDP(..., bucket_cap_mb=25)` vs 默认 25MB vs 100MB——看 step time 变化

### 成功标准
- 给出 N=1/2/4（有 8 卡就到 8）的 (tokens/sec, tokens/sec/GPU, scaling_efficiency) 表
- efficiency 预期（**取决于互联，标注你的硬件**）：
  - **PCIe**（g4dn 4×T4 / g5 4×A10G）：4 卡 ≥ 60%
  - **NVLink/NVSwitch**（p3 V100 / p4d A100）：8 卡 ≥ 80%
  - 不达标不算失败——baby GPT 太小、backward 太快、all-reduce 来不及 overlap 也会压低 efficiency，写进 README 分析即可
- 能解释为什么 efficiency < 1：通信开销没完全和 backward overlap

### 失败排查
- **Efficiency 突降到 50% 以下**：
  - bucket size 太小 → 通信次数多、overlap 不充分
  - dataloader 卡了 → 用 A-M0 任务 2 的方法调 num_workers
  - 模型太小 → backward 太快、来不及和 all-reduce overlap（这种情况换大模型比调 DDP 有效）
- **DDP 报 `find_unused_parameters` 错误**：模型有些参数本 step 没参与 forward → 加 `find_unused_parameters=True`，但要知道这有性能代价（本模块过关题里有问）
- **rank 0 跑得正常，其他 rank 卡住**：probably 同步点不一致——保证所有 rank 走同样的 forward / backward，不要在某个 rank 单独 print/save 阻塞别人

### 辅助阅读（非 canonical）
- PyTorch `DistributedSampler` 文档：https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
- nanoGPT 的 `train.py` DDP 部分（参考写法，**别抄**）：https://github.com/karpathy/nanoGPT/blob/master/train.py

### Deliverable
- `course-a/m2-ddp/train_ddp.py`（在 mygpt 包外，import mygpt 用）
- `course-a/m2-ddp/scaling_results.md`：硬件 + 表格 + 一段解释

---

## 任务 3 · Gradient accumulation（动手）+ no_sync（📖 读懂）

### 为什么做这个
显存装不下你想要的大 batch，是训练里天天遇到的现实约束。梯度累加就是绕过这个约束的标准手法——用"小批量多跑几次、把梯度攒起来再更新"换来等效的大 batch，代价是慢一点。亲手跑几组不同的拆分方式，你能直接看到"micro-batch 越大越喂得满 GPU、但显存也越吃紧"这个 trade-off 长什么样，从此对"显存不够时该怎么办"有真实的操作经验，而不是只知道有这么个名词。

### 动手部分：accum 的显存↔吞吐 trade-off
这部分值得跑——「小 micro-batch 多次累加」vs「大 batch 一次」的显存/吞吐差异是实测才有体感的。

1. 加 `--accum_steps` 和 `--micro_batch` flag，实现 accum 循环（贴合本项目 `get_batch` + `criterion`）：
   ```python
   for step in range(num_steps):
       optimizer.zero_grad()                    # 在外层，绝不放进内层
       for i in range(accum_steps):
           x, y = get_batch(train_data, micro_batch, block_size, local_rank)
           output = model(x)
           B, T, C = output.shape
           loss = criterion(output.view(B*T, C), y.view(B*T)) / accum_steps  # 除 accum_steps
           loss.backward()
       optimizer.step()
   ```
2. 固定 global batch（4 卡时 micro × accum = 64），跑 (micro=1,accum=64) / (4,16) / (16,4) / (64,1)，记 (step time, peak memory)

**成功标准**：看到 micro ↑ → step time ↓（少 accum 次、GPU 喂得满）+ peak memory ↑（activation ∝ batch）。这个 trade-off 就是 grad accum 的存在意义——显存不够放大 batch 时，用小 micro + 多 accum 换时间。

> 注：你已经跑出 `accum_result.md`（1227ms/7.2GB → 321ms/13.5GB），动手部分 ✅ 完成。

### 📖 读懂部分（不需动手，QUIZ 自查）：`no_sync`
- **机制一句话**：DDP 默认每次 `backward()` 都触发 all-reduce；grad accum 时前 `accum_steps-1` 次梯度还没累加完、同步没意义。`model.no_sync()` 上下文跳过这些 step 的 all-reduce，只在最后一次同步 → accum_steps 次 backward 只 1 次通信。
- **为什么不必动手验证**：NVLink 上省的通信在 step time 上几乎看不出（你 8 卡 efficiency 已 94%），PCIe/跨机才显著。机制读懂 + QUIZ 能答即可，不值得为它造对照实验。
- QUIZ 自查：「no_sync 是干什么的？不用会怎样？」能答清 = 过。

---

## 任务 4 · Mixed precision（📖 读懂，不需动手）

### 为什么做这个
现在几乎所有大模型训练都跑在半精度（bf16/fp16）上，因为它能省一半显存、还能用上 tensor core 提速——不懂这块，你连别人怎么把大模型塞进显卡的都看不明白。这里有几个反复被问的关键点：为什么 fp16 必须配 GradScaler 而 bf16 不用、混精度到底省了哪块显存又有哪块没省。这些是绕不开的基础概念，所以即使不动手，也得读到能讲清楚的程度。

> **降级理由**：fp16/bf16/GradScaler 是一句话能讲清的概念，动手跑三组数字对理解无增量。而且你在 V100 上没有 bf16 tensor core，对比本就残缺。**读懂 + QUIZ 自查即可，不要求写代码、不产出 result.md**。

读懂这几点（QUIZ 会考）：
- **autocast**：把 forward/loss 里的 matmul 等跑成半精度（省显存、提速），数值敏感的 op（softmax/norm/loss reduce）自动保 fp32。
- **fp16 vs bf16**：fp16 = 5 exp + 10 mantissa，范围窄（max ~65504）易 underflow/overflow；bf16 = 8 exp + 7 mantissa，范围和 fp32 一样宽但精度低。
- **GradScaler**：**fp16 必须用**——梯度容易 underflow 成 0，scaler 先把 loss 乘大常数、step 前再除回，把梯度拉进 fp16 可表示范围。**bf16 不用**（范围够宽不会 underflow）。
- **省的是什么**：activation + 半精度副本省显存；matmul 走 tensor core 提速。optimizer state（AdamW m/v）默认仍 fp32（精度敏感）——这是 A-M3 ZeRO 才会分掉的部分。
- **硬件**：bf16 需 A100/H100 tensor core；V100/T4 只能 fp16。

QUIZ 自查：「fp16 为什么需要 GradScaler、bf16 为什么不需要？」「mixed precision 省了哪块显存、哪块没省？」

---

## 任务 5 · `torch.profiler` 看 overlap（📖 读懂，不需动手）

### 为什么做这个
DDP 之所以能做到接近线性加速，靠的是"边算梯度边把算好的梯度发出去"——让通信藏在计算后面，而不是算完再傻等着同步。profiler 就是能让你在时间线上亲眼看到这种重叠的工具。这里的核心结论（为什么梯度同步发生在反向传播阶段、怎么验证通信和计算是重叠的）你在任务 2 的吞吐数据里其实已经间接验证过了，所以这里只需读懂工具用法和这个结论，真正需要动手抓 trace 是后面跑大模型做瓶颈分析时的事。

> **降级理由**：profiler 是工具、用法看一眼教程就会；「DDP 在 backward overlap all-reduce」这个结论你在任务 2 的 scaling efficiency 里已经实测过了（step time N=1→N≥2 只多 0.008s 就是 overlap 的证据）。专门抓 trace 截图对理解无增量。**读懂即可，不要求产出 trace 截图**。

读懂这几点：
- **怎么用**：`torch.profiler.profile(activities=[CPU, CUDA], schedule=..., on_trace_ready=tensorboard_trace_handler('./log'))`，只 profile 中间几个 step（不然 trace 文件几 GB）。用 tensorboard 或 chrome://tracing 看 timeline。
- **要看什么**：backward 阶段穿插的 `ncclAllReduce`——某层 grad 算完立刻 all-reduce，同时下一层 backward 继续 → 通信藏在计算后面（overlap）。
- **为什么 DDP 在 backward 触发 all-reduce 而非 forward**：梯度是 backward 算出来的；放 backward 还能和后续层的 backward overlap，把通信时间藏住。这是 A-M2 任务 2 你 94% efficiency 的原因。

QUIZ 自查：「DDP 为什么在 backward 触发 all-reduce？怎么验证 NCCL 和 backward 是 overlap 的？」（你任务 2 的 step time 数据就是答案）

> 真到 A-M4 跑 1B 时会用 profiler 做 bottleneck 归因（那是真需要动手的场景）——A-M2 这里读懂用法就够。

---

## A-M2 做完之后

- **动手已完成**：bench_collectives（任务 1）+ scaling（任务 2）+ grad accum trade-off（任务 3 动手部分）
- **读懂 + QUIZ 自查**：no_sync / mixed precision / profiler
- 跑 QUIZ.md，上面几个概念能答清 = 过关 → 开 A-M3（FSDP / ZeRO）
