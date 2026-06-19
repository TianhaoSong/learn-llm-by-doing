# A-M4 · Doing 任务详细规格

> Topic：把 A-M0 到 A-M3 串成一次真实长训练——在多 GPU 上用真实 web 数据把 ~350M 模型训过 ≥1B tokens，训练脚本内置 sharded checkpoint + 中断 resume，中途抓 profiler 做 bottleneck 归因。线 1 终点。
>
> - **环境**：`ml.p3dn.24xlarge`（8× V100 32GB, NVLink），同 A-M3。
> - **规模**：~350M 模型（同 A-M3），训过 ≥1B tokens。「1B」是 token 数，不是参数量。
> - **精度**：fp32（同 A-M3）。V100 无 bf16 tensor core；350M fp32 在 32GB 上装得下（A-M3 已验证 zero3 ~8.6G），不上混精度。

---

## 为什么做这个
TinyShakespeare 那点数据训出来的模型只会背莎士比亚；真正的语言模型是在海量真实网页文本上训出来的。这个任务把前面学的所有东西（模型、DDP、FSDP、显存账本）第一次拧成一次真实长训练：在真 web 数据上跑过 10 亿 token。长训练随时可能挂（机器故障、spot 回收），所以脚本必须内置 checkpoint/resume——一中断能从上次接着训、loss 不断档。跑的过程中用 profiler 看时间花在哪（算 / 通信 / 等数据），定位瓶颈、并预判模型放大后瓶颈怎么变。这套"长跑 + 存档续训 + 性能归因"是真实训大模型的完整闭环，也是线 1 的终点和实证。

## 目标
一次长跑：~350M 模型 + 真实 web 数据，训过 ≥1B tokens，loss 收敛、能续写连贯英文。脚本内置 sharded checkpoint/resume，中途产出一份 bottleneck 报告。

---

## 组件 1 · 数据：FineWeb tokenize 落盘
- 装 `pip install datasets`
- 写 `prepare.py`：streaming load `HuggingFaceFW/fineweb-edu` 的 `sample-10BT` 子集，用 tiktoken gpt2 encode，token 落 `numpy.uint16` 二进制 `train.bin` / `val.bin`
- 参考 nanoGPT 的 prepare.py：https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
- 落盘大小：≥1B tokens × 2 bytes = ≥2 GB，放 EBS gp3
- **不要 on-the-fly tokenize**——CPU tokenize 跟不上 GPU，dataloader 会成瓶颈。先 tokenize 一次落盘，训练时 `np.memmap` 读

## 组件 2 · 训练脚本 `train_large.py`（长跑 + ckpt/resume + MFU，不含 profiler）
基于 A-M3 的 `fsdp_train.py` 扩展（FSDP zero3 + fp32），用 `np.memmap` 读 `train.bin`。建模型/FSDP wrap 抽到 `model.py`、sampler 抽到 `data.py`（profile_run.py 也 import 这俩，保证两边模型一致）：
- sampler：每次 forward 随机抽 `micro_batch` 个起点、各取 `block_size` 长度。**每个 rank 必须抽不同位置**（按 rank 错开随机种子）——否则 8 张卡算出相同梯度，all-reduce 后 effective batch 退化成 `micro_batch × grad_accum`（32），白瞎了 ×8。memmap 随机采样不像 `DistributedSampler` 自动按 rank 切，得手动保证
- lr schedule（**本课程第一次用，下面讲清**）：分两段——
  - **warmup**：训练头若干步，lr 从 0 **线性爬到目标 lr**。为什么：刚初始化的权重 + Adam 的 m/v（动量/方差估计）还没攒够统计，此时直接上大 lr 容易一脚踩崩（loss 飞 / NaN）。先小 lr 走稳、统计攒够，再上全速。
  - **cosine decay**：warmup 到顶后，lr 沿余弦曲线**从目标值平滑降到 ~0**。为什么：后期接近收敛，需要小 lr 精调，否则在最优点附近来回跳；cosine 形状前缓后陡，比线性 decay 经验上更好，是 GPT/LLaMA 预训练标配。
  - **warmup 步数取总步数 ~2%、且封顶 10%**，别写死大数（本 run 才 ~4000 步，写死 1000 = 25% 在热身，离谱）：`warmup = min(total_steps // 10, max(20, int(total_steps * 0.02)))`——total=4000 → 80；total=50 → 5。
  - 不用手推 cosine 公式，用现成的 `transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)`（见辅助阅读）。
- grad clip：`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`（防 NaN）

**具体参数（8× V100 32GB, 350M, zero3, fp32, seq=1024）——必须靠 grad accumulation 凑 batch，不能直接开大 micro-batch**：

| 参数 | 值 | 来源 |
|---|---|---|
| `micro_batch`（每卡每次 forward） | **2** | A-M3 实测：zero3 batch=2/GPU peak ~8.6G，32G 有余量。直接开大（如 256）必 OOM——activation 才是瓶颈 |
| GPU 数 | 8 | p3dn |
| `grad_accum` | 16 | 攒梯度凑大 effective batch，不增显存 |
| **effective batch** | 8 × 2 × 16 = **256 sequences** | 全局,每张卡负责 256/8 = 32 条（= micro_batch 2 × accum 16），但显存里一次只 2 条。= 256 × 1024 ≈ **26 万 token/step** |
| step 数 | ~4000 | **由 token 预算除出来的**：1.05B ÷ 26万/step ≈ 4000，不是"4000 步就练好了"的收敛点。A-M4 的终止条件是跑够 ≥1B tokens（验证工程链路），不以达到某 loss 为目标 |

> 关键：**effective batch 256 是靠 `8 卡 × micro_batch 2 × grad_accum 16` 凑出来的，不是每卡一次塞 256**。per-GPU 一次只跑 2 条序列（A-M3 实测能跑），grad accum 在不加显存的前提下把有效 batch 放大到 256。想更快可试 micro_batch=4（zero3 下约 +5G，仍 < 32G），相应把 grad_accum 减半保持 effective=256。

**内置 sharded checkpoint + resume**（长跑必须）。自己写，用 `torch.distributed.checkpoint`（DCP）。

**策略：主循环里每 N 步（如 500）存一次，重启从最近的 ckpt 接着跑。** 定期 save 本来就在所有 rank 的同步点上，不用碰 signal/SIGTERM 那套——spot 被收走最多丢 < N 步（几分钟），对 350M 规模完全可接受，比"一步不丢"的信号方案简单得多、也不会写错。

要点和坑：
- **save/load 是集合操作，所有 rank 必须一起调**——不是 rank0 单独存。每个 rank 写/读自己那片，所以快、且能 reshard（8 卡存的换 4 卡也能 load）。
- **model/optimizer state 怎么拿**：FSDP 下不要手动 `model.state_dict()`，用 DCP 配套的 `get_state_dict` / `set_state_dict`（`torch.distributed.checkpoint.state_dict`）——它们负责把分片 state 转成可存/可载的形式。
- **step、lr_scheduler、dataloader 位置这些标量 DCP 不管**（它只存 tensor state）——存进 ckpt metadata，load 后自己读出来 set 回去。resume 后 loss 跳变，十有八九是这些没接上（见失败排查）。

→ 完整 API 用法见 [PyTorch Distributed Checkpoint tutorial](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)。

**MFU 监控**（看实际用了多少 GPU 算力）：
- 理论峰值：**V100 fp32 ≈ 15.7 TFLOPS**（你这台）。注：A100 bf16 是 312、H100 bf16 990——精度/卡不同峰值差很多，**MFU 数字只在同精度同卡间可比**，别拿 fp32 的跟 bf16 的比
- 实际 FLOPs/step ≈ `6 × n_params × tokens_per_step`（fwd+bwd 共 6N；开 activation ckpt 重算变 ~8N）
- MFU = 实际 FLOPs/s ÷ 理论峰值
- 每 N step 打 `(step, val_loss, tokens/sec, MFU, peak_mem)`，记本地 jsonl 或 wandb

## 组件 3 · 一次性 profiling 诊断 → bottleneck 报告

**profiling 是一次性诊断，不是训练的一部分。** 它跟 ckpt/MFU 这些"训练内在环节"性质不同：你跑**一次**看清瓶颈在哪，结论拿到就完事——瓶颈结构（compute/NCCL/dataloader 占比）由模型架构 + 精度 + 硬件决定，前几百步就稳定，跟跑 110 步还是 1B tokens 无关。所以**不要把 profiler 塞进 `train_large.py`**（那会引入 mode flag、trace overhead 污染长跑的 MFU/tok-s、两次跑共用 ckpt/日志互相覆盖等一堆问题）。

**做法：单独写 `profile_run.py`**，`import` 和 train_large.py 同样的 model / data / FSDP 构建函数（所以两边模型完全一致），但它只跑 ~110 步、只抓 trace、不存 ckpt、不写训练日志：
```python
# profile_run.py —— 独立脚本，只做诊断这一件事
from model import build_model      # 与 train_large.py 共用
from data import make_sampler
...
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=98, warmup=2, active=10),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace"),
) as prof:
    for step in range(110):
        train_step(); prof.step()
```
train_large.py 保持纯粹——只管长跑 + 定期 ckpt，**完全不知道 profiler 存在**。两个脚本各干一件事、不共享任何输出，所以怎么跑都不会互相污染。

抓完用 tensorboard / chrome://tracing 看 timeline（trace 文件可能上 GB），把时间分四类量化（占比直接看 TensorBoard 的 Execution Summary 表，或自己解析 trace.json 累加各类 kernel 时长）：
- **Compute**：forward + backward 的 GPU kernel
- **NCCL**：`ncclAllGather` / `ncclReduceScatter`（FSDP 主要这俩）——**必须 8 卡下跑才有**，单卡 profile 这块恒为 0
- **Dataloader stall**：timeline 里 GPU 干等数据的空白（GPU util 高、idle 低 = 无 stall）
- **Other**：optimizer step、H2D copy（memcpy）、memset 等

写 `bottleneck_report.md`：四类占比表 + 当前瓶颈判断 + 一段"模型再大 4 倍瓶颈会变成什么"的推理（通常 NCCL：all-gather buffer 翻倍但带宽不变；或 activation 翻倍触发更频繁 ckpt 重算）。

---

## 验证：resume 不断档
- 把 save 间隔临时设小（如每 100 step），跑到 100 step 存一次
- 直接 `kill -9` 杀进程（模拟 spot 被收走），从最近 ckpt resume 继续跑一两百步
- **判据：看 resume 接缝处的 loss 曲线，应平滑接着降，没有 spike / 台阶**——这一次 resume run 自己就能看出来，不用另跑一次完整 baseline 对照
- 为什么 spike 是判据：Adam 的 m/v（optimizer state）若没恢复、被重置成 0，resume 后头几十步 loss 会明显跳起再慢慢降（等于重新热身）；params 没恢复则直接崩。**别拿单步 loss 比"差几个百分点"**——每步抽到哪批随机数据 loss 本就上下抖，单步差异是噪声、说明不了恢复对没对，要看的是接缝有没有结构性的凸起
- 想更严格：把数据采样按 step 设成确定性（同 step 抽同一批），resume 后对应 step 的 loss 应近乎逐位一致——但这要额外恢复 RNG/采样状态，非必须

## 成功标准
- 模型能续写连贯英文（不追 perplexity，loss 单调下降趋于平稳即可）
- 训过 ≥1B tokens，训练日志（step / val_loss / tokens/sec / MFU / peak_mem）全程完整
- resume 测试：kill 后从 ckpt 续跑，接缝处 loss 平滑无 spike
- bottleneck 报告：四类占比表（和 ~100%）+ 当前瓶颈 + 再大 4 倍的预测
- MFU：fp32 在 V100 上本就偏低，能跑出一个稳定数字 + 解释瓶颈即可（不卡具体阈值）

## 失败排查
- **dataloader 是瓶颈**（profiler 看到大段 GPU idle）：np.memmap 一次性 mmap 别每步重 open；加 dataloader worker
- **跑到一半 NaN**：lr 太大 / warmup 不够 / 漏了 grad clip
- **resume 后 loss 跳变**：optimizer state 没 load 上（AdamW 的 m/v 重置成 0）；或 lr scheduler 的 step 没 resume；或 dataloader 没从对应位置接上
- **ckpt load 报错**：save/load 的 FSDP API 版本要一致（都用 `torch.distributed.checkpoint`）
- **S3 写超时**：`s3fs` 没装 / 权限不够；先用本地路径跑通再切 S3

## 辅助阅读（非 canonical）
- Karpathy "Reproducing GPT-2 in llm.c"（MFU 计算 + 真实 web 训练）：https://github.com/karpathy/llm.c/discussions/677
- HuggingFace FineWeb dataset card：https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- PyTorch Distributed Checkpoint tutorial（Getting Started with DCP，含 get_state_dict/set_state_dict 示例）：https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
- PyTorch profiler recipe：https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- HuggingFace optimizer schedules（lr warmup + cosine 现成实现）：https://huggingface.co/docs/transformers/main_classes/optimizer_schedules

## Deliverable
- `course-a/m4-large-run/prepare.py`（FineWeb tokenize 落盘）
- `course-a/m4-large-run/model.py` + `data.py`（建模型/FSDP wrap + memmap sampler，train 和 profile 共用）
- `course-a/m4-large-run/train_large.py`（FSDP fp32 长跑 + 每 N 步 ckpt/resume + MFU 监控；不含 profiler）
- `course-a/m4-large-run/profile_run.py`（一次性诊断：import model/data，跑 ~110 步抓 trace，不存 ckpt）
- `course-a/m4-large-run/training_log.{md,jsonl}` 或 wandb 链接
- `course-a/m4-large-run/sample_output.txt`（最终模型续写 ~500 token）
- `course-a/m4-large-run/bottleneck_report.md`（profiler 四类占比 + 瓶颈判断 + 4× 预测）

> ckpt 大小自查：sharded ckpt 每 rank ≈ `(fp32 params 4B + AdamW m/v 8B) × n_params / N`（≈ 12 字节/参数 ÷ N），不含训练时的 activation/bf16 副本——ckpt 只存能恢复训练的状态。

---

## 这个任务做完之后
- 跑 QUIZ.md
- A-M4 过关 → **课程 A 完成** → 开 B-M0（推理 baseline）
