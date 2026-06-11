# A-M4 · Doing 任务详细规格

> Topic：把 A-M0 到 A-M3 串起来——AWS 多 GPU 真实数据 ≥ 1B tokens 训练 + sharded checkpoint + spot 中断 resume + profiler bottleneck 归因。线 1 终点。
>
> 四个任务：资源申请 / 数据 pipeline + 训练 / checkpoint resume / bottleneck 报告。

---

## 任务 1 · GPU 资源申请

### 为什么做这个
前面所有东西都是在小卡或少量卡上跑通的，但真要训一个 1B 规模的模型，你得先拿到一批真正的多卡 A100/H100。申请 GPU 配额、配好访问、验证 NCCL 能正常跑通，这套流程本身就是真实工作里训大模型的第一道坎——往往比写代码还磨人。把它走一遍并记下踩到的坑，是后面三个任务能真正开跑的前提；同时坚持用最小权限、不碰生产，也是该养成的安全习惯。

### 目标
申请 ≥ 4 卡 A100/H100（理想 8 卡 p4d/p4de），ReadOnly / 最小权限，不动生产。

### Sub-tasks
1. 查你的 GPU 平台的 onboarding 文档
2. 申请流程通常包括：
   - 提 ticket 走 quota approval（一般 1-2 周）
   - 拿到 cluster ID + IAM role（用 ReadOnly 最小权限版本，不要 Admin）
   - 配 SSH / SageMaker training job——选其一就行
3. 验证连接：在分配的 instance 上跑 `nvidia-smi` 看 GPU 数；跑 `torch.cuda.device_count()` 在 Python 里确认；跑你 A-M2 的 `bench_collectives.py` 确认 NCCL 配置正确
4. **如果 GPU 资源 onboarding 太慢（> 2 周）**：fallback 用个人 AWS spot p4d（~$10/hr spot）跑通；技术上等价，钱自己出

### 成功标准
- 拿到 ≥ 4 卡 A100/H100 的稳定 access
- `bench_collectives.py` 在该 cluster 上跑出来的数字与 A-M2 一致量级
- 写一份 onboarding 笔记（自己用，记下踩到的合规 / 网络 / 存储坑）

### 辅助阅读
- 资源具体路径查对应平台文档，不要硬编码 URL
- AWS Spot Best Practices（fallback 用）：https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html

### Deliverable
- `course-a/m4-large-run/onboarding_notes.md`（私人笔记，不要 commit 任何资源链接 / cluster ID）

---

## 任务 2 · FineWeb tokenize 落盘 + ≥ 1B tokens 训练

### 为什么做这个
TinyShakespeare 那点数据训出来的模型只会背莎士比亚，真正的语言模型是在海量真实网页文本上训出来的。这个任务把前面学的所有东西（模型、DDP、FSDP、混精度、显存账本）第一次拧成一个真实训练任务，在真 web 数据上跑过 10 亿 token——这是整条训练线的终点和实证。过程中你还要监控 MFU，也就是你实际用到了多少 GPU 算力，这个数字直接告诉你训练效率高不高，是衡量"跑得好不好"而不只是"跑通了"的核心指标。

### 目标
用真实 web 数据（不是 TinyShakespeare）训练 ≥ 100M（推荐 350M-1B）模型。这是 A 课程的最大单次训练，也是面试讲"我跑了 1B tokens"的实证。

### Sub-tasks

1. **数据准备：FineWeb-Edu 子集 tokenize 落盘**
   - 装 `pip install datasets`
   - 写 `prepare.py`：streaming load `HuggingFaceFW/fineweb-edu` 的 `sample-10BT` 子集，用 tiktoken gpt2 encode，token 落 `numpy.uint16` 二进制文件 `train.bin` / `val.bin`
   - 参考 nanoGPT 的 prepare.py 模式：https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
   - 落盘大小：≥ 1B tokens × 2 bytes = ≥ 2 GB；放 EBS gp3
   - **不要** on-the-fly tokenize——dataloader 会成为瓶颈

2. **训练脚本 `train_large.py`**：基于 A-M3 的 train_fsdp.py，扩到 350M 或 1B；用 `np.memmap` 读 train.bin
   - sampler：每 step 随机抽 batch_size 个起点拿 block_size 长度
   - lr schedule：warmup 1000 step + cosine decay
   - 跑 ≥ 1B tokens（如 batch=256, block=1024, step=4000 = ~1B tokens）

3. **MFU（Model FLOPs Utilization）监控**：
   - 算理论峰值 FLOPs：A100 bf16 = 312 TFLOPS, H100 bf16 = 990 TFLOPS
   - 算实际 FLOPs：每 step 大约 `6 × n_params × tokens_per_step`（forward + backward 共 6N，激活 checkpointing 加倍变 8N）
   - MFU = actual / theoretical / time
   - 目标：30-50% MFU；< 20% 说明有问题

4. **训练日志**：每 N step 打 `(step, val_loss, tokens_per_sec, MFU, peak_mem)`；建议同时记到 wandb 或本地 jsonl 文件

5. **结束训练后 generate 一段样例**：用最终 checkpoint 续写一个 prompt，看输出是否连贯

### 成功标准
- 模型能续写连贯英文（不追 perplexity，loss 单调下降到 ~3.0 以下并趋于平稳）
- ≥ 1B tokens 训练完成
- MFU ≥ 30%（A100 bf16 + FSDP + 1B 模型 + 8 卡）
- 训练日志完整（step / val_loss / MFU 全程记录）

### 失败排查
- **dataloader 是瓶颈**：np.memmap 读得太慢 → 多 worker 并行 fetch；或者 batch 取的位置经常 cache miss → 把 train.bin 一次性 mmap 不要每步重 open
- **MFU < 15%**：通信占比太高（profiler 看）；或者 batch 太小、forward/backward 太短；放大 micro-batch
- **跑到一半 NaN**：lr 太大 / lr warmup 不够 / 没 grad clip（加 `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`）
- **跨卡 loss 抖动巨大**：DistributedSampler 的 `set_epoch` 漏了（每个 epoch shuffle 顺序不一致）

### 辅助阅读（非 canonical）
- Andrej Karpathy "Reproducing GPT-2 (124M) in llm.c"（讲 MFU 计算和真实 web 训练）：https://github.com/karpathy/llm.c/discussions/677
- HuggingFace FineWeb dataset card：https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

### Deliverable
- `course-a/m4-large-run/train_large.py`
- `course-a/m4-large-run/prepare.py`
- `course-a/m4-large-run/training_log.md` 或 wandb 链接
- `course-a/m4-large-run/sample_output.txt`：最终模型续写 ~500 token

---

## 任务 3 · Sharded checkpoint + spot 中断 resume

### 为什么做这个
大模型一训就是几天甚至几周，机器随时可能挂——尤其用便宜的 spot 实例，云厂商提前两分钟通知就把卡收走了。如果不会存档续训，一中断就得从头再来，前面烧的钱和时间全打水漂。这个任务让你学会"被打断也能从上次的地方接着训、loss 曲线不断档"：每张卡只存自己那一片参数所以又快又省，收到中断信号就赶紧存一次再退出。这是真实训大模型时绕不开的保命技能，也是面试常被追问的点。

### 目标
实现"训 100 step → save → 重启 → 继续训 100 step，loss 曲线连续"的端到端 resume。这是 spot 训练的必备技能，也是面试常问点。

### Sub-tasks

1. **Sharded checkpoint save**：
   ```python
   import torch.distributed.checkpoint as dcp
   from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
   model_state, optim_state = get_state_dict(model, optimizer)
   state = {"model": model_state, "optimizer": optim_state, "step": step, "val_loss": last_val_loss}
   dcp.save(state, checkpoint_id=f"s3://your-bucket/ckpt/step-{step}")  # 或本地路径
   ```
   每 rank 只写自己的 shard、并行写、快 N 倍

2. **Sharded checkpoint load**：
   ```python
   state = {"model": model_state, "optimizer": optim_state, "step": 0, "val_loss": 0}
   dcp.load(state, checkpoint_id=f"s3://your-bucket/ckpt/step-{ckpt_step}")
   set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
   step = state["step"]
   ```
   data loader 也要 resume——记下 step 后从对应位置继续 sample

3. **Spot 中断处理**：
   ```python
   import signal
   def handler(signum, frame):
       save_checkpoint()
       sys.exit(0)
   signal.signal(signal.SIGTERM, handler)
   ```
   AWS spot 中断给 2 分钟通知发 SIGTERM；catch 到就触发 save 然后 exit；下次 launch 自动从最新 ckpt resume

4. **Resume 测试**：
   - 跑 100 step 触发 save，记 `loss_at_100`
   - 杀进程
   - 重启从 ckpt resume，继续 100 step（共 200 step）
   - 验证：第 101 step 开始的 loss 与一气跑完的同 step loss 几乎一致（差异 < 5%）

### 成功标准
- save / load 跑通；ckpt 文件大小合理（每 rank ~ params/N × 6 字节）
- Resume 测试：100→200 step 的 loss 曲线在 ckpt 切点平滑过渡
- spot 中断 handler 测试：手动 `kill -TERM <pid>` 验证 save 触发

### 失败排查
- **load 后 loss 跳变（不连续）**：optimizer state 没 load 上（AdamW 的 m/v 重置成 0）；或者 lr scheduler step 没 resume
- **load 后 NaN**：FSDP1 用 SHARDED_STATE_DICT save 的没法在 FSDP2 load；保持 API 一致
- **S3 写超时**：`s3fs` 没装 / IAM 权限不够；先用本地路径跑通再切 S3

### 辅助阅读（非 canonical）
- PyTorch Distributed Checkpoint tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
- AWS Spot Instance Interruption Notice: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html

### Deliverable
- 在 `train_large.py` 内集成 save/load + signal handler
- `course-a/m4-large-run/resume_test.md`：贴 100 / 200 step 的 loss 数字证明连续

---

## 任务 4 · Profiler bottleneck 报告

### 为什么做这个
训练慢、MFU 上不去时，光靠猜没用——你得知道时间到底花在哪：是 GPU 在算、在卡间传数据、还是在干等着读数据。profiler 抓一段时间线就能把这几块的占比量化出来，让"瓶颈在哪"从拍脑袋变成有据可查。更重要的是学会往前推一步：现在不是瓶颈的东西，模型放大几倍后可能就成了瓶颈（比如通信量翻倍但带宽没变）。这种"看数据定位瓶颈、再预判下一个瓶颈"的能力，是从"能把训练跑起来"进阶到"能把训练调好"的关键，面试里聊性能优化也必然会问到。

### 目标
用 `torch.profiler` 抓一段 trace（step 100-110），把训练时间分成 (compute / NCCL / dataloader stall / other) 四块，写报告说明哪一块是当前瓶颈、再大 4 倍模型瓶颈会变成什么。

### Sub-tasks

1. 在 train_large.py 加 profiler context（schedule 跑 step 100-110）
2. 抓完 trace 用 tensorboard 或 chrome://tracing 看 timeline
3. 量化：
   - **Compute 占比** = forward + backward 的 GPU kernel 时间
   - **NCCL 占比** = `ncclAllReduce` / `ncclAllGather` / `ncclReduceScatter` 的 GPU 时间（FSDP 主要是 all-gather + reduce-scatter）
   - **Dataloader stall** = 看是否有 GPU idle 等数据的 gap（在 timeline 里的空白）
   - **Other** = optimizer step、数据 H2D copy、其他
4. 写 `bottleneck_report.md`：
   - 一张时间分布表（4 类的 % + 总时间）
   - 一段判断：当前瓶颈是哪类
   - 一段预测："如果模型再大 4 倍，瓶颈会变成什么"——通常是 NCCL（all-gather buffer 翻 4 倍但带宽不变；或 activation 翻 4 倍 OOM 触发更频繁的 checkpoint 重算）

### 成功标准
- 4 类占比表给全（加起来 ~100%）
- bottleneck 判断有 trace 截图佐证
- "再大 4 倍" 的预测有合理的推理路径，不是凭感觉

### 失败排查
- **compute 占比 > 90%**：恭喜，你的训练是 compute-bound（最理想状态）；这种情况 MFU 应该 > 40%
- **dataloader stall 大段空白**：np.memmap 读得慢 / num_workers 太少；用 A-M0 任务 2 的方法调
- **NCCL 占比 > 50%**：FSDP 没和 forward overlap；或者 bucket 太小；或者 PCIe 互联本来就慢

### 辅助阅读（非 canonical）
- HOLISTIC TRACE ANALYSIS（PyTorch 官方分布式 trace 工具）：https://github.com/facebookresearch/HolisticTraceAnalysis
- "How to read a PyTorch profiler trace" 博客（搜 Meta / PyTorch 官方文章）

### Deliverable
- `course-a/m4-large-run/bottleneck_report.md`（含 trace 截图 + 4 类占比 + 当前瓶颈判断 + 4× 预测）

---

## 四个任务做完之后

- 跑 QUIZ.md
- A-M4 过关 → 课程 A 完成 → 开 B-M0（推理 baseline）
