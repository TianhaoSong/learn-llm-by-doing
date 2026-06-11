# learn-llm-by-doing — Course Outlines

> 目的：把 `requirements.md` 的三条工作线拆成可执行的模块化课程。
> 每个模块独立可过关，串起来就是面试可讲的完整故事。
> 所有 Out-of-Scope 项（chatbot/RAG、深度数学推导、大段博客、死磕调参）都不作为学习目标。

## 动手 vs 读懂（doing 任务的取舍原则）

**doing 任务只覆盖两类**——其他一律走 QUIZ 自查、不要求动手：
1. **手写核心机制**：transformer / KV cache / PagedAttention / tensor parallel / DDP / agent loop —— 不自己写一遍建立不了理解。
2. **实测才有体感的数字**：通信带宽 / scaling efficiency / 显存账本 / TTFT-TPOT / 吞吐 —— 不跑出来没概念。

**一句话能讲清的概念，知道就行、不动手**（动手无理解增量，反而陷入抠代码）：如 `no_sync`（前 N-1 次 backward 跳过 all-reduce）、mixed precision（fp16 要 GradScaler、bf16 不用）、profiler 用法、跨机互联差一个数量级。这些标注「📖 读懂 / QUIZ 自查」，不要求写代码、不要求产出 result.md。

> 判据：问自己「不动手写，光读懂能不能在 QUIZ 答清楚 + 面试讲明白？」能 → 读懂就行；不能（机制藏在代码里 / 数字要实测）→ 动手。

## 💰 全项目 AWS 成本预算（纯 spot 自费）

| 模块 | 实例 | 估算 | 省钱关键 |
|---|---|---|---|
| A-M0 / A-M1 | g4dn.xlarge / g5.xlarge（单卡） | ~$7 | 单卡，跑完即关 |
| A-M2 | g4dn.12xlarge / p3.8xlarge（4卡） | ~$15 | T4/V100 够，不用 A100 |
| A-M3 | g5.12xlarge / p4d（4–8卡） | ~$20 | 与 A-M4 连做省冷启动 |
| **A-M4** | p4d（8× A100）**跑 350M 不跑 1B** | ~$40 | 最大杠杆：350M 而非 1B（省 ~$110） |
| B-M0~M2 | g5.xlarge（单卡 + GPT-2） | ~$10 | 调度逻辑不吃大卡 |
| B-M3 + B-M4 | p4d（8卡）**连做** | ~$60 | 一次开机做完两个 7B 推理模块 |
| C 课程 | t3.medium + API token | ~$50 | 成本全在 API，机器几乎免费 |
| **合计** | | **~$200** | |

**三条铁律**：① 代码先在单卡/小实例调通，多卡只在跑 benchmark 时开、**跑完立刻 terminate**；② spot 必配 checkpoint（被回收不丢进度）+ auto-shutdown（忘关一晚 p4d ≈ $200）；③ **成本大头（A-M3/A-M4/B-M3/B-M4）优先用 AWS GPU（spot 或已有 quota）**，自费可压到 < $80。

---

## 课程 A — 手写训练（线 1）

**Course outcome（学完后能讲清楚）**：从 nanoGPT 单卡跑通到 100M–1B 多卡训练，能脱口讲清 DDP/FSDP/ZeRO 的取舍、NCCL 通信模式、显存账本、gradient accumulation 与 micro-batch 的 trade-off，并有自己的 profiling 数据。

### A-M0 · PyTorch & 训练循环基础

**Topic**：建立 PyTorch 单 GPU 训练的全栈心智模型——autograd 计算图、CPU→GPU dataloader pipeline、GPU kernel 异步执行。三件事串起来构成"为什么我的训练能跑、跑得快不快、出 bug 怎么定位"的最小完整闭环。后续 A-M1（nanoGPT）/ A-M2（DDP）都假设这一层已经内化。

0. **环境准备**

   **快速 setup**（命令照抄）：
   - 实例：`g4dn.xlarge` spot（1× T4，最省 ~$0.2/hr）；MNIST + cuda_stream demo 单卡足够
   - AMI：AWS Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04) — 已含 CUDA + PyTorch
   - 装包：`pip install matplotlib pandas`（dataloader bench 画图用）
   - 数据：MNIST 由 `torchvision.datasets.MNIST(root='./data', download=True)` 自动下载
   - 仓库布局：`mkdir -p course-a/m0-pytorch && cd $_`
   - 启动：`python train_mlp.py`（单卡，无需 torchrun）
   - 💰 **成本纪律**：跑完 benchmark/实验立刻 `terminate`，别挂机；单卡模块 < $2

   **值得理解**（看材料、问自己）：
   - DLAMI 把 CUDA driver / PyTorch / cuDNN 版本一致性问题搞定了——后续所有线 1/2 模块复用同一个 AMI，省掉 90% env 折腾
   - 启动时打一行 `torch.cuda.is_available() / torch.cuda.device_count()` 自检，避免无声 fall back 到 CPU
   - `cuda_stream_demo.py` 的 timing 必须用 `torch.cuda.Event` + `event.synchronize()`，不是 `time.time()`——CPU timer 看不到 GPU 异步执行

1. **学习目标**
   - 能独立写 `nn.Module` / training loop / optimizer step / lr scheduler / dataloader
   - 理解 autograd：何时 `.detach()` / `.no_grad()` / `retain_graph`
   - 区分 device/host 内存、`pin_memory`、`non_blocking` 的作用
   - CUDA 执行模型基础：stream 是什么、kernel launch 异步性、`cudaStreamSynchronize` / `cudaEvent` 何时必需（**只用 PyTorch API 体感即可，不要求写 CUDA kernel**）

2. **学习材料（canonical）**
   - PyTorch 60-min Blitz: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
   - PyTorch Autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html
   - Karpathy "The spelled-out intro to neural networks and backpropagation: building micrograd": https://www.youtube.com/watch?v=VMj-3S1tku0
   - PyTorch CUDA semantics（重点读 stream / async / synchronization 三节）: https://pytorch.org/docs/stable/notes/cuda.html
   - NVIDIA "How to Overlap Data Transfers in CUDA C/C++"（概念读物，不要求写 C++）: https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/

   > 这里只列**核心**——读完直接对得上学习目标。任务相关的辅助阅读（如 dataloader 性能调优）放在 `DOING.md` 内引用。

3. **Doing 任务** → 见 [`course-a/m0-pytorch/DOING.md`](course-a/m0-pytorch/DOING.md)
   - 三个任务：`train_mlp.py` / `bench_dataloader.py` / `cuda_stream_demo.py`
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-a/m0-pytorch/QUIZ.md`](course-a/m0-pytorch/QUIZ.md)
   - 口试题用于自查知识是否吃透（不是面试演练）
   - 任务级 code review checkpoint 与 benchmark 数字写在 DOING.md（"做完"的定义），QUIZ.md 只放概念性问答

> 注：mixed precision（fp16/bf16/GradScaler）推到 A-M2，等多卡训练显存有真实压力时再学。

---

### A-M1 · Transformer & nanoGPT 单卡复现

**Topic**：把 transformer 从"读过 paper"变成"能徒手写出来"——GPT-2 small 结构、causal attention、TinyShakespeare 单卡跑通、与参考实现数值对照。后续 A-M2 多卡训练和 B 课程推理引擎都依赖这一层结构。

0. **环境准备**

   **快速 setup**：
   - 实例：`g5.xlarge` spot（GPT-124M + TinyShakespeare 单卡够）；想跑快点用 `g5.2xlarge`（A10G 24GB）
   - AMI：同 A-M0
   - 装包：`pip install tiktoken datasets wandb`（wandb 可选）
   - 数据：TinyShakespeare 一键下载 `wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
   - 仓库布局：`mkdir -p course-a/m1-mygpt/{mygpt,tests}`
   - 启动：`python -m mygpt.train`

   **值得理解**：
   - **tokenizer 选 `tiktoken` 的 `gpt2` encoding**——直接复用 OpenAI BPE，不用自己训；`tiktoken.get_encoding("gpt2")` 一行搞定
   - **数值对照测试**用 `F.scaled_dot_product_attention` 当 ground truth；它走 PyTorch 内置的 fused kernel（FlashAttention/CuDNN/math 之一），数值精度可信
   - **`torch.compile` 第一次跑会编译 30s+**——bench step time 时丢掉前几步 warmup
   - **不用 nanoGPT 的 train.py**：你要从零写。但 `model.py` 卡住时可以对照 nanoGPT 看是否结构错位

1. **学习目标**
   - 能徒手写出 multi-head self-attention（含 causal mask）、LayerNorm/RMSNorm、residual、FFN
   - 解释 RoPE / 绝对位置编码 / ALiBi 的差异
   - 解释 BPE/byte-level tokenizer 的工作流程（不要求自己训 tokenizer）
   - 知道 KV cache 在推理时长什么样（为线 2 埋伏笔）

2. **学习材料（canonical）**
   - "Attention Is All You Need": https://arxiv.org/abs/1706.03762
   - Karpathy "Let's build GPT: from scratch, in code, spelled out": https://www.youtube.com/watch?v=kCc8FmEb1nY
   - nanoGPT repo: https://github.com/karpathy/nanoGPT
   - The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
   - RoFormer (RoPE) paper: https://arxiv.org/abs/2104.09864

   > 任务相关辅助阅读（tiktoken 用法、`torch.compile` 注意点等）放 DOING.md。

3. **Doing 任务** → 见 [`course-a/m1-mygpt/DOING.md`](course-a/m1-mygpt/DOING.md)
   - 三个任务：`mygpt/model.py + train.py` / `tests/test_attention.py` / `torch.compile` step time 实验
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-a/m1-mygpt/QUIZ.md`](course-a/m1-mygpt/QUIZ.md)
   - 任务级 deliverable / benchmark 数字 / code review checkpoint 在 DOING.md
   - QUIZ.md 是 transformer 内部机制 + 训练 / 推理时 KV cache 形状 + 数值稳定性的概念自查

---

### A-M2 · 单机多卡 DDP + Mixed Precision

**Topic**：从单卡跨到多卡——NCCL collective ops 实测带宽、DDP 内部机制（bucket / overlap）、gradient accumulation 与 micro-batch 的 trade-off、mixed precision 的数值与显存收益。这一层是面试"多 GPU 训练"的核心入口，也是 A-M3 FSDP 的前提。

0. **环境准备**

   **快速 setup**：
   - 实例（按省优先）：`g4dn.12xlarge` spot（4× T4，PCIe，最便宜的 4 卡）跑 DDP 概念 + scaling 曲线完全够；想看 **NVLink overlap** 选 `p3.8xlarge` spot（4× V100 NVLink，常比 g5 还便宜）——baby GPT 不需要 A100，**p4d 留给 A-M3**
   - AMI：同 A-M0
   - 装包：无新增（DLAMI 自带 NCCL + torch.distributed）
   - 数据：复用 A-M1 的 TinyShakespeare 即可（不用 OpenWebText，太快收敛看不出 scaling 问题；DDP 重点是 throughput 和 scaling，不是模型质量）
   - 仓库布局：`mkdir -p course-a/m2-ddp`，复用 A-M1 的 mygpt 包
   - 启动：`torchrun --standalone --nproc_per_node=4 train_ddp.py`
   - 💰 **成本纪律**：代码先在单卡/本地调通，**多卡只在跑 benchmark 时开、跑完立刻 terminate**；A-M2 整个 < $15

   **值得理解**：
   - **`torchrun --nproc_per_node=N` 实际 fork 出 N 个 Python 进程**，每进程占一张 GPU；`RANK` / `LOCAL_RANK` / `WORLD_SIZE` 通过环境变量传入。这是面试题。
   - **`init_process_group(backend="nccl")` 在每个进程里调一次**；它不是创建进程，只是让已 fork 出的进程互相 rendezvous
   - **NCCL 调试三神器**：`NCCL_DEBUG=INFO`（看 transport：NVLink / PCIe / IB）、`NCCL_DEBUG_SUBSYS=ALL`、`TORCH_DISTRIBUTED_DEBUG=DETAIL`。卡 hang 时先打开这些
   - **g5 是 PCIe 互联（无 NVLink）**——你跑出来的 DDP scaling 比 p4d 差是预期，bench 里要标注硬件
   - **profiler 抓 trace** 用 `torch.profiler.profile(activities=[CPU, CUDA], on_trace_ready=tensorboard_trace_handler('./log'))`，输出文件用 `chrome://tracing/` 或 `tensorboard --logdir=./log` 看 timeline
   - **bf16 vs fp16**：A100/H100 都原生支持 bf16，几乎默认选 bf16（无需 GradScaler）；fp16 主要是 V100/T4 这类老卡

1. **学习目标**
   - NCCL collective ops 语义：all-reduce / reduce-scatter / all-gather / broadcast / barrier
   - DDP 的工作机制：bucket、overlap backward 与 gradient all-reduce、`find_unused_parameters` 的代价
   - `torchrun` / `init_process_group` / rank / world_size / local_rank
   - gradient accumulation 与 micro-batch 的关系，DDP 下如何正确累积（`no_sync` 用法）
   - mixed precision（fp16/bf16）和 GradScaler 的工作原理；为什么多卡训练几乎一定要开

2. **学习材料（canonical）**
   - PyTorch DDP tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
   - PyTorch DDP design note: https://pytorch.org/docs/stable/notes/ddp.html
   - NVIDIA NCCL docs (collective ops): https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
   - "PyTorch Distributed: Experiences on Accelerating Data Parallel Training": https://arxiv.org/abs/2006.15704
   - PyTorch AMP recipe: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
   - "Mixed Precision Training" (Micikevicius et al.): https://arxiv.org/abs/1710.03740
   - HuggingFace "Methods and tools for efficient training on a single GPU / multiple GPUs": https://huggingface.co/docs/transformers/perf_train_gpu_many

   > 任务相关辅助阅读（nccl-tests、profiler 用法、bucket size 调优等）放 DOING.md。

3. **Doing 任务** → 见 [`course-a/m2-ddp/DOING.md`](course-a/m2-ddp/DOING.md)
   - 五个任务：`bench_collectives.py` / DDP 版 mygpt / gradient accumulation / mixed precision / profiler trace
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-a/m2-ddp/QUIZ.md`](course-a/m2-ddp/QUIZ.md)
   - 任务级 benchmark 数字 / scaling 曲线 / peak memory 表格在 DOING.md
   - QUIZ.md 是 NCCL 通信模式 + DDP 内部机制 + grad accumulation 数学 + 数值精度的概念自查

---

### A-M3 · 显存账本与 ZeRO/FSDP 渐进升级

**Topic**：训练显存的精确账本（params / grads / optimizer state / activations 四块字节数）+ ZeRO Stage 1/2/3 把哪三块分给哪个 rank + FSDP 是 ZeRO 在 PyTorch 里的实现 + activation checkpointing 的 compute↔memory trade-off。能从"DDP 跑不动 350M"升级到"FSDP 在同硬件上跑得动"并量化每一步的显存收益。

0. **环境准备**

   **快速 setup**：
   - 实例（按省优先）：`g5.12xlarge` spot（4× A10G 24GB）或 `p3.8xlarge`（4× V100 16GB）就够演示 DDP→FSDP 渐进——**V100 16GB 上 350M 的 DDP baseline 会 OOM，这恰好是"DDP 跑不动、FSDP 跑得动"的教学场景**。`p4d.24xlarge`（8× A100）仅在你想 **A-M3 + A-M4 连做**（同一台跑完显存对比直接进 1B 训练）时才开，省一次冷启动
   - AMI：同 A-M0
   - 装包：无新增；FSDP 是 PyTorch 原生
   - 数据：用 OpenWebText 子集 `load_dataset("Skylion007/openwebtext", split="train[:1%]")` 或 FineWeb-Edu 子集——350M 模型需要更多 token 数据才看得出显存差异
   - 仓库布局：`mkdir -p course-a/m3-fsdp`
   - 启动：`torchrun --standalone --nproc_per_node=4 train_fsdp.py --shard_strategy=FULL_SHARD`
   - 💰 **成本纪律**：代码单卡调通 → 多卡只跑显存/throughput 对比 → 跑完 terminate；4 卡几小时 < $20。**优先用 AWS GPU（spot 或已有 quota）**（免费，requirements.md 写了你有）

   **值得理解**：
   - **PyTorch 2.x 有两套 FSDP API**：`torch.distributed.fsdp.FullyShardedDataParallel`（FSDP1，老）vs `torch.distributed._composable.fsdp.fully_shard`（FSDP2，新）。**新项目用 FSDP2**——API 更简洁、和 TP 组合更顺；但材料里还是 FSDP1 多，读的时候注意分辨
   - **显存测量必须用 `torch.cuda.reset_peak_memory_stats()` 清零再跑一段 + `torch.cuda.max_memory_allocated()`**；`memory_summary()` 打的是当前快照，不是 peak
   - **`memory._record_memory_history(enabled='all')` + `_dump_snapshot('out.pickle')`** 然后用 https://pytorch.org/memory_viz 在线 viewer 打开——能看到 allocator 每一笔 alloc/free 的来源，调显存爆炸神器
   - **FSDP `auto_wrap_policy`**：默认不 wrap 会把整个模型当一个 shard 单元，等于退化成 ZeRO-3 但通信粒度极差。**必须用 `transformer_auto_wrap_policy`** 按 transformer block wrap
   - **activation checkpointing** 用 `torch.utils.checkpoint.checkpoint`（手动）或 `apply_activation_checkpointing`（FSDP 配套）；它的代价是 backward 重算 forward——所以 throughput 会降 20–30%

1. **学习目标**
   - 能列出训练显存四大块：parameters / gradients / optimizer states / activations，并对 fp32/mixed-precision/AdamW 计算字节数
   - 解释 ZeRO Stage 1/2/3 各分什么、通信代价分别是什么
   - 解释 FSDP（PyTorch 原生）与 DeepSpeed ZeRO 的对应关系
   - activation checkpointing 的 trade-off（compute ↔ memory）

2. **学习材料（canonical）**
   - "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models": https://arxiv.org/abs/1910.02054
   - PyTorch FSDP tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
   - PyTorch FSDP advanced: https://pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html
   - DeepSpeed ZeRO docs: https://www.deepspeed.ai/tutorials/zero/
   - "Reducing Activation Recomputation in Large Transformer Models": https://arxiv.org/abs/2205.05198
   - HuggingFace memory anatomy: https://huggingface.co/docs/transformers/model_memory_anatomy

   > 任务相关辅助阅读（activations 字节数公式推导、FSDP2 vs FSDP1、memory_viz 工具用法等）放 DOING.md。

3. **Doing 任务** → 见 [`course-a/m3-fsdp/DOING.md`](course-a/m3-fsdp/DOING.md)
   - 三个任务：`mem_calc.py` 显存账本计算 + 实测验证 / DDP→FSDP 渐进升级 + activation checkpointing / 显存收益归因表
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-a/m3-fsdp/QUIZ.md`](course-a/m3-fsdp/QUIZ.md)
   - 任务级 benchmark 四元组 / 显存账本表格在 DOING.md
   - QUIZ.md 是 ZeRO Stage 通信差异 + FSDP 实现细节 + activation checkpointing 数学的概念自查

---

### A-M4 · 100M–1B 多卡训练实跑（线 1 终点）

**Topic**：把 A-M0 到 A-M3 学到的所有东西串起来——在 AWS 多 GPU 上跑通 ≥ 100M 参数模型的 FSDP 训练 + 真实数据 pipeline（FineWeb tokenize 落盘）+ sharded checkpoint + spot 中断 resume + profiler bottleneck 归因。这一模块的产出（完整训练日志 + bottleneck 分析）是线 1 在面试中可讲的最高密度证据。

0. **环境准备**

   **快速 setup**：
   - **默认跑 350M，不跑 1B**（最大省钱杠杆）：350M + FSDP 在 `p4d.24xlarge`（8× A100 40GB）就够，~$40 跑完；1B 要 `p4de`（80GB）+ 翻倍时长 ~$150。**学习目标（多卡 FSDP + checkpoint resume + bottleneck 分析）350M 和 1B 完全一样**，按 requirements.md「模型质量不是目标」，350M 足够。想跑 1B 再升 p4de
   - 💰 **资源来源优先级**：① AWS GPU（spot 或已有 quota）→ ② AWS spot p4d。**A-M3 用 p4d 的话，A-M3+A-M4 连着做、同一台机器跑完**，省冷启动 + onboarding
   - 💰 **spot 纪律**：sharded checkpoint（本任务核心）+ 脚本末尾 auto `shutdown` / 设 `--max-runtime`；**忘关一晚 p4d ≈ $200 没了**，这是最常见烧钱方式
   - AMI：同 A-M0
   - 数据：FineWeb-Edu 推荐——更新、质量比 OpenWebText 高
     ```
     pip install datasets
     python -c "from datasets import load_dataset; ds=load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True); ..."
     ```
     先 streaming 下载 + tokenize 落到 `.bin` 文件（参考 nanoGPT 的 prepare.py 模式），训练时 mmap 读
   - 存储：tokenized 数据 ~10–50GB 放 EBS gp3；checkpoint 写 S3（spot 中断后能跨实例 resume）
   - 仓库布局：`mkdir -p course-a/m4-large-run`
   - 启动：`torchrun --standalone --nproc_per_node=8 train_large.py --resume_from=s3://your-bucket/ckpts/`

   **值得理解**：
   - **数据 tokenize 落盘 vs on-the-fly**：on-the-fly tokenize 会让 dataloader 成为瓶颈（CPU 跟不上 GPU），所以**先 tokenize 一次落 `.bin`，训练时 numpy.memmap 读**——这是 nanoGPT 的标准做法，A-M4 必须这么做
   - **spot 中断处理**：监听 `SIGTERM`（spot 给 2 分钟通知），catch 到就触发 sharded ckpt save → exit；下次 launch 自动从 S3 最新 ckpt resume。这正是过关任务的"100 step → save → 重启 → loss 连续"
   - **sharded checkpoint API**（FSDP2）：`torch.distributed.checkpoint.save / load` + `DCP.FileSystemWriter('s3://...' )`（需要 `s3fs`）。每个 rank 只写自己的分片，速度快 N 倍
   - **profiler 在长跑里要采样**：跑 step 100–110 这一段开 profiler 就够，一直开会爆 trace 文件几 GB
   - **ReadOnly / 最小权限**（合规）：跑训练只需要 S3 read/write 自己的 bucket + EC2 launch；**绝不用 Admin role**，按 requirements.md 项目内合规约束
   - **MFU（Model FLOPs Utilization）目标 30–50%**——p4de + bf16 + FSDP 跑 1B 拿 ~30% MFU 是合理基线，低于 20% 说明 dataloader 或通信卡了

1. **学习目标**
   - 在 AWS 多 GPU 资源上跑通 ≥ 100M 参数模型的 FSDP 训练
   - 能用 profiler 定位 bottleneck（compute-bound vs comm-bound vs IO-bound）
   - 能解释 checkpoint 保存/恢复在 FSDP 下的注意点（`StateDictType.SHARDED_STATE_DICT` vs `FULL_STATE_DICT`）

2. **学习材料（canonical）**
   - PyTorch profiler tutorial: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
   - "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM": https://arxiv.org/abs/2104.04473（只读 §2-3，理解 3D parallelism 概念，不要求实现）
   - FSDP checkpoint API: https://pytorch.org/docs/stable/distributed.checkpoint.html
   - AWS GPU 资源 onboarding（查你的 GPU 平台 / SageMaker 文档）

   > 任务相关辅助阅读（FineWeb tokenize 落盘、spot 中断处理、MFU 计算等）放 DOING.md。

3. **Doing 任务** → 见 [`course-a/m4-large-run/DOING.md`](course-a/m4-large-run/DOING.md)
   - 四个任务：GPU 资源申请 / FineWeb tokenize 落盘 + ≥ 1B tokens 训练 / sharded checkpoint resume 测试 / profiler bottleneck 报告
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-a/m4-large-run/QUIZ.md`](course-a/m4-large-run/QUIZ.md)
   - 任务级完成清单（模型续写样例 + 训练日志 + checkpoint resume 验证 + bottleneck 报告）在 DOING.md
   - QUIZ.md 是 sharded checkpoint 设计 + bottleneck 归因方法 + MFU 计算 + spot 中断处理的概念自查

---

## 课程 B — 简化版 vLLM（线 2）

**Course outcome（学完后能讲清楚）**：从 naive batched inference 到手写 PagedAttention + tensor parallel，每个阶段都有 throughput/latency benchmark；能脱口讲清 KV cache 设计、continuous batching 的调度策略、prefill vs decode 的资源差异，并能与真实 vLLM 对比说出差距与原因。

> 前置：课程 A 的 M1（Transformer + KV cache 直觉）已完成。

### B-M0 · 推理基础与 baseline

**Topic**：建立 LLM 推理的基础心智模型——prefill (compute-bound) vs decode (memory-bound) 的资源差异、TTFT / TPOT / 吞吐这些 serving 指标的精确定义、各种 sampling 算法。这一层是后续 B-M1 (KV cache + continuous batching) 到 B-M4 (与 vLLM 对比) 所有优化工作的对照基线。

0. **环境准备**

   **快速 setup**：
   - 实例：`g5.xlarge` spot（A10G 24GB，跑 GPT-2 small 完全够；后续 B-M3/B-M4 升 `p4d`/`p4de`）
   - AMI：同 A-M0
   - 装包：`pip install transformers accelerate sentencepiece`
   - 模型：`from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained("gpt2")`（首次 ~500MB，缓存到 `~/.cache/huggingface`）
   - 仓库布局：`mkdir -p course-b/m0-baseline`
   - 启动：`python naive_infer.py`

   **值得理解**：
   - **TTFT vs TPOT 测量必须分开计时**：`prefill_end = time.time()` 在第一个 token 产出后；后面每个 token decode 时间单独计。**整体 latency / total_tokens 是错的指标**
   - **`torch.no_grad()` + `model.eval()` 都要打**——`no_grad` 关 autograd 省显存；`eval()` 关 dropout/BN training mode
   - **bf16 推理**：A10G 原生支持 bf16，`model.to(dtype=torch.bfloat16)` 一行；fp16 在老卡上更快但有溢出风险
   - **bench 时 warmup 5–10 步丢掉**——CUDA kernel 第一次 launch 有编译/加载开销；不 warmup 测的是 cold start

1. **学习目标**
   - 区分 prefill（compute-bound, batched matmul）与 decode（memory-bound, KV-cache reads）
   - 理解 LLM serving 的关键指标：TTFT、TPOT、吞吐 tokens/sec、p50/p99 latency
   - sampling 基础：greedy / temperature / top-k / top-p / repetition penalty

2. **学习材料（canonical）**
   - "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM paper): https://arxiv.org/abs/2309.06180
   - HuggingFace generation strategies: https://huggingface.co/docs/transformers/main/en/generation_strategies
   - "A Survey on Efficient Inference for Large Language Models": https://arxiv.org/abs/2404.14294
   - Lilian Weng "Large Transformer Model Inference Optimization": https://lilianweng.github.io/posts/2023-01-10-inference-optimization/

   > 任务相关辅助阅读（HF transformers `generate` 内部细节、bf16 推理、profiler 测 prefill/decode 等）放 DOING.md。

3. **Doing 任务** → 见 [`course-b/m0-baseline/DOING.md`](course-b/m0-baseline/DOING.md)
   - 三个任务：`naive_infer.py` baseline + TTFT/TPOT 测量 / sampling 实现与单测 / `baseline_bench.md` 对照基线
   - 模型权重来源（GPT-2 / 自训 / Llama-2-7B 等）的取舍在 DOING.md 说明
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-b/m0-baseline/QUIZ.md`](course-b/m0-baseline/QUIZ.md)
   - 任务级 benchmark 数字（baseline 表）在 DOING.md
   - QUIZ.md 是 prefill vs decode 资源差异 + serving 指标定义 + sampling 数学的概念自查

---

### B-M1 · 手写 KV cache + continuous batching

**Topic**：自己写 KV cache 数据结构 + continuous batching 调度器，理解为什么 static batching 会有 padding waste 和 head-of-line blocking、continuous batching 怎么用 iteration-level scheduling 解决。这一层是面试"continuous batching 调度策略"的核心。

0. **环境准备**

   **快速 setup**：
   - 实例：`g5.xlarge` spot 仍然够（B-M1 重点是调度逻辑，不是大模型）
   - 装包：`pip install asyncio aiohttp`（如果做 async API；不做就纯 Python 单进程也行）
   - 数据：自己合成 stress test workload——`prompts = [random_prompt(length=randint(50, 2048)) for _ in range(1000)]`
   - 仓库布局：`mkdir -p course-b/m1-engine/engine_v1`
   - 启动：`python -m engine_v1.engine`（暴露 `add_request` API，主循环跑 stress test）

   **值得理解**：
   - **KV cache 形状**：`[num_layers, 2, batch, num_heads, max_seq_len, head_dim]`，2 是 K/V。**预分配 max_seq_len 一次，每步只在 seq 维 append**——避免每步 `torch.cat` 触发新 alloc
   - **Continuous batching scheduler 状态机**（核心实现）：每个 request 的状态 = `WAITING / PREFILL / DECODE / FINISHED`；每个 iteration 调度器检查队列，把能塞下的 PREFILL 加入 batch，已在 DECODE 的继续 decode，FINISHED 的释放 KV slot
   - **prefill 和 decode 不能简单 concat 进同一 batch**：prefill 的 attention mask 是 lower triangular，decode 是单 token 看全 cache。**两条路**：(a) 同 batch 但分开 forward（两次 kernel 调用）；(b) 用变长 attention（FlashAttention 风格的 `cu_seqlens`）合并——B-M1 推荐 (a)，B-M2 之后再考虑 (b)
   - **不要用 `torch.cat` append KV**——会触发新 alloc + memcpy。用预分配 tensor + 写入对应 slot：`kv_cache[layer, 0, batch_idx, :, current_pos, :] = new_k`

1. **学习目标**
   - 实现一个 KV cache：形状、何时分配、何时释放、如何与 attention 接口对齐
   - continuous batching（也叫 in-flight batching / iteration-level scheduling）：与 static batching 的差异，调度循环长什么样
   - 解释 head-of-line blocking 和 padding waste 在 static batching 下的危害

2. **学习材料（canonical）**
   - "Orca: A Distributed Serving System for Transformer-Based Generative Models" (continuous batching 起源): https://www.usenix.org/conference/osdi22/presentation/yu
   - Anyscale "How continuous batching enables 23x throughput in LLM inference": https://www.anyscale.com/blog/continuous-batching-llm-inference
   - vLLM scheduler source code（带着读，不要求完全模仿）: https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py
   - HuggingFace TGI continuous batching docs: https://huggingface.co/docs/text-generation-inference/conceptual/streaming

   > 任务相关辅助阅读（KV cache shape 推导、prefill+decode 合并 batch 的实现细节）放 DOING.md。

3. **Doing 任务** → 见 [`course-b/m1-engine/DOING.md`](course-b/m1-engine/DOING.md)
   - 三个任务：`engine_v1/`（kv_cache + scheduler + engine 主循环） / stress test / `cb_vs_static.md` 对照
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-b/m1-engine/QUIZ.md`](course-b/m1-engine/QUIZ.md)
   - 任务级 benchmark 数字（吞吐 ≥ 2× baseline 等）+ code review checkpoint 在 DOING.md
   - QUIZ.md 是 KV cache 数据结构 + 调度状态机 + prefill/decode 资源争抢的概念自查

---

### B-M2 · 手写 PagedAttention

**Topic**：理解为什么 B-M1 的连续 KV cache 在变长输出下浪费显存（fragmentation + over-provisioning），用 block-based 分页方案（PagedAttention）让显存利用率上去。这一层是面试"为什么需要 PagedAttention" 的核心。注意：**本模块不以 throughput 为过关指标**——纯 PyTorch 的 paged attention 单 op 比连续 cache 慢 30-50% 是预期，throughput 优化要等到 vLLM 的 CUDA kernel；过关只看显存利用率与最大并发请求数。

0. **环境准备**

   **快速 setup**：
   - 实例：`g5.xlarge` 够；想看显存对比效果选 `g5.2xlarge`（24GB 单卡）跑稍大模型
   - 装包：无新增；选做的 Triton kernel 需要 `pip install triton`（DLAMI 通常已装）
   - 仓库布局：`mkdir -p course-b/m2-paged/engine_v2`
   - 启动：`python -m engine_v2.engine`

   **值得理解**：
   - **block 物理布局**：所有 block 是一个大 tensor `physical_blocks[total_blocks, num_layers, 2, num_heads, block_size, head_dim]`；free list 是 `set[int]` 维护可用 block id；逻辑映射 `block_table[seq_id] = [block_id_0, block_id_1, ...]`
   - **PyTorch 实现 paged attention** 用 `torch.gather` 或 `torch.index_select` 把每个 seq 的 block 拼回连续 tensor 再做 attention——这就是慢的来源（gather 每步触发数据搬运）。**vLLM 的 CUDA kernel 直接在 block 上算，不拼回**
   - **block size 选 16**：太小（如 4）→ block table 大、index 开销高；太大（如 64）→ fragmentation 回来了。vLLM 默认 16 是经验最优
   - **OOM 处理**：调度器在 prefill 一个新请求需要 N 个 block 但 free list 不够时，要么 (a) 拒绝该请求（简单），要么 (b) preempt 已有 DECODE 请求（vLLM 做法，把 KV swap 到 CPU）。**B-M2 实现 (a) 即可**，(b) 是高级话题
   - **selectively 看 vLLM 源码**：`vllm/core/block_manager.py` 和 `vllm/attention/ops/paged_attn.py`——读懂数据结构和 kernel launch 入口即可，不要陷入 CUDA 细节

1. **学习目标**
   - 解释为什么连续 KV cache 在变长输出下会浪费显存（fragmentation + over-provisioning）
   - PagedAttention 数据结构：block table、physical block、logical-to-physical 映射
   - copy-on-write 与 prefix caching 的关系（理解即可，不强求实现 prefix caching）
   - 解释 attention kernel 在 paged 布局下需要做什么改动

2. **学习材料（canonical）**
   - vLLM paper 重读 §4-5（数据结构与 kernel）: https://arxiv.org/abs/2309.06180
   - vLLM PagedAttention kernel C++/CUDA 入口（读懂 host 侧 launch 代码即可）: https://github.com/vllm-project/vllm/tree/main/csrc/attention
   - Aleksa Gordić "PagedAttention from scratch" 系列博客（如可访问）: https://github.com/gordicaleksa/get_started_with_open_source_llms

   > 任务相关辅助阅读（block size 选择经验、`torch.gather` 实现细节、Triton kernel 入门）放 DOING.md。

3. **Doing 任务** → 见 [`course-b/m2-paged/DOING.md`](course-b/m2-paged/DOING.md)
   - 三个任务：`engine_v2/`（block_manager + paged_attention 集成调度器）/ 显存效率对比 / 选做 Triton kernel
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-b/m2-paged/QUIZ.md`](course-b/m2-paged/QUIZ.md)
   - 任务级显存数字（max concurrent requests ≥ 1.5× B-M1 等）+ code review checkpoint 在 DOING.md
   - QUIZ.md 是 fragmentation 来源 + block table 设计 + copy-on-write + 显存收益 vs 单 op 开销分离的概念自查

---

### B-M3 · Tensor parallel + 多机部署

**Topic**：tensor parallel 在 attention（沿 head 维切）和 FFN（column→row 组合）上的切法 + 通信模式（每层 1 次 all-reduce）+ NVLink/EFA/TCP 拓扑对 TP 效率的影响 + 推理 TP vs 训练 FSDP 的取舍。这一层是面试"为什么训练用 FSDP / 推理用 TP" 的核心答案。

0. **环境准备**

   **快速 setup**：
   - 单机：`p4d.24xlarge`（8× A100 40GB NVLink）跑 Llama-2-7B TP=2/4/8 够；优先 spot（或已有 quota）
   - 跨机（选做）：2× `p4de.24xlarge` 同 placement group + EFA 启用
   - 模型：切到 Llama-2-7B（HF）——TP 数字才好看；GPT-2 small 太小切了反而慢
     ```
     huggingface-cli login  # 接受 Llama license
     huggingface-cli download meta-llama/Llama-2-7b-hf
     ```
   - 仓库布局：`mkdir -p course-b/m3-tp/engine_v3`
   - 启动：`torchrun --standalone --nproc_per_node=8 -m engine_v3.engine`（单机 TP=8）
   - 💰 **成本纪律**：TP 代码逻辑先在小模型/少卡验证对 → 真跑 7B benchmark 时才开 8 卡 → 跑完 terminate。**B-M3 + B-M4 连着做**（同一台 p4d、都是 7B 推理，一次开机做完两个模块）；两个模块合计 < $60

   **值得理解**：
   - **PyTorch Tensor Parallel API**（`torch.distributed.tensor.parallel`）：`ColwiseParallel` / `RowwiseParallel` 是新一代 API，比手写 all-reduce 简洁；但**手写一遍**对理解最有帮助——这是面试题
   - **column-parallel 后接 row-parallel**：`Y = (X @ W_col)` 各 rank 持有部分列结果，**接下来 row-parallel matmul 自带的 all-reduce** 把它合回——这就是 Megatron 论文的 trick，attention 和 FFN 都靠这个组合**全程只有 1 次 all-reduce/层**
   - **QKV 切法**：把 Q/K/V projection 合并成一个大矩阵 column-parallel 切，每 rank 拿走 `(num_heads / TP)` 个 head——这是为什么 num_heads 必须能被 TP 整除
   - **EFA 跨机配置（选做）**：实例必须在 cluster placement group 内、AMI 含 EFA driver、IAM 允许 EFA、`NCCL_PROTO=Simple` + `FI_PROVIDER=efa`。**任何一项配错都退化成 TCP，跨机带宽掉到 1/10**
   - **拿不到跨机也能讲清楚**：NVLink ~600 GB/s，EFA ~100 GB/s，TCP ~10 GB/s——一个数量级差。面试问"TP=8 跨机比单机慢多少"答"all-reduce 的关键路径慢一个数量级" 就够

1. **学习目标**
   - tensor parallel 在 attention/FFN 上的切法（column-parallel 与 row-parallel 的搭配）
   - tensor parallel 的通信模式（forward all-reduce vs reduce-scatter + all-gather）与 latency 影响
   - 多机部署下的 NCCL 拓扑（NVLink intra-node、IB/EFA inter-node）对 TP 效率的影响
   - 与课程 A 的 FSDP 对比：训练 vs 推理的并行需求差异

2. **学习材料（canonical）**
   - "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism": https://arxiv.org/abs/1909.08053
   - vLLM distributed inference docs: https://docs.vllm.ai/en/latest/serving/distributed_serving.html
   - PyTorch Tensor Parallel API: https://pytorch.org/tutorials/intermediate/TP_tutorial.html
   - SGLang architecture overview（对照阅读，不需要复现）: https://github.com/sgl-project/sglang

   > 任务相关辅助阅读（Megatron column/row parallel 推导、EFA 配置、NVLink vs PCIe 带宽对比）放 DOING.md。

3. **Doing 任务** → 见 [`course-b/m3-tp/DOING.md`](course-b/m3-tp/DOING.md)
   - 三个任务：`engine_v3/` TP 实现（手写 column/row parallel） / 单机 TP 扩展曲线 / 跨机 TP 测试 + 报告
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-b/m3-tp/QUIZ.md`](course-b/m3-tp/QUIZ.md)
   - 任务级 benchmark 数字（TP 扩展曲线 + 跨机 latency）+ code review checkpoint 在 DOING.md
   - QUIZ.md 是 column/row parallel 数学 + 通信拓扑 + 训练 FSDP vs 推理 TP 取舍的概念自查

---

### B-M4 · 与真实 vLLM/SGLang 对比（线 2 终点）

**Topic**：把自己的 engine（B-M0 → B-M3 拼起来）和 vLLM / SGLang 在同硬件 / 同模型 / 同 workload 下跑、把差距逐项归因到具体机制（FlashAttention / CUDA graph / prefix caching / 量化），给出"如果继续投入会先做什么"的优先级清单。这是线 2 终点，也是面试讲推理深度的最高密度证据。

0. **环境准备**

   **快速 setup**：
   - 实例：`p4d.24xlarge`（8× A100 40GB）跑 Llama-2-7B；优先 spot（或已有 quota）。**接着 B-M3 同一台机器做**（都是 7B 推理，别重新开）；想跑 13B 升 `p4de`
   - 装包：
     ```
     pip install vllm  # 注意 CUDA 版本必须匹配 DLAMI 的；通常 pip install 自动选对
     pip install sglang[all]
     ```
   - 数据：ShareGPT 子集 https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered（注意：这是真实对话数据，仅供 benchmark）
   - 仓库布局：`mkdir -p course-b/m4-compare`
   - 启动：分别启 self-engine / vLLM serve / SGLang serve，三个进程各占一段时间窗口跑同一组 prompt
   - 💰 **成本纪律**：三方 benchmark 一气跑完（self / vLLM / SGLang 错峰但同一开机窗口）→ 跑完 terminate；与 B-M3 合计 < $60

   **值得理解**：
   - **vLLM 启动**：`python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --tensor-parallel-size 8`，暴露 OpenAI 兼容 API
   - **bench 脚本**：vLLM 自带 `benchmarks/benchmark_serving.py` ——直接用它跑你的 self-engine（实现 OpenAI API 接口）和 vLLM/SGLang，三方数据可比
   - **归因清单**（重要）：差距来自 (a) FlashAttention kernel（vLLM 用，你没用，差 2–3×），(b) CUDA graph（vLLM 用 `enforce_eager=False`，省 kernel launch，差 10–20%），(c) prefix caching（vLLM v0.4+ 默认开），(d) 量化（你 bf16，vLLM 可 fp8/awq）。逐项归因 = B-M4 的核心交付
   - **同硬件、同模型、同 workload** 是公平对比的关键——任何一个不同结论就不可信
   - **不要纠结绝对差距**——3× 慢是正常的，PyTorch naive 实现对比生产级系统差 3–5× 都合理。面试讲的是"差距来自哪里"而不是"我多快"

1. **学习目标**
   - 在相同硬件、相同 workload 下用自己实现的 engine 与 vLLM、SGLang 对比
   - 能逐项归因差距：kernel 优化（FlashAttention、CUDA graph）、调度策略、量化、prefix cache 等
   - 给出一个"如果继续投入会先做什么"的优先级清单

2. **学习材料（canonical）**
   - vLLM benchmarks scripts: https://github.com/vllm-project/vllm/tree/main/benchmarks
   - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness": https://arxiv.org/abs/2205.14135（只读 §1-3，理解为什么是 IO-bound 优化，不要求实现）
   - SGLang RadixAttention paper: https://arxiv.org/abs/2312.07104
   - NVIDIA CUDA graph + LLM serving 介绍: https://developer.nvidia.com/blog/accelerating-llms-with-cuda-graphs/

   > 任务相关辅助阅读（OpenAI API spec、vLLM/SGLang 启动配置、ShareGPT 数据格式）放 DOING.md。

3. **Doing 任务** → 见 [`course-b/m4-compare/DOING.md`](course-b/m4-compare/DOING.md)
   - 三个任务：self-engine 暴露 OpenAI API + vLLM/SGLang 起服务 / 三方 benchmark 对比 / `self_vs_vllm.md` 归因 + `next_steps_priorities.md`
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-b/m4-compare/QUIZ.md`](course-b/m4-compare/QUIZ.md)
   - 任务级 benchmark 表 + 归因清单 + 优先级清单在 DOING.md
   - QUIZ.md 是 FlashAttention IO-aware 思想 + CUDA graph + RadixAttention + 量化的概念自查

---

## 课程 C — Agent

**Course outcome（学完后能讲清楚）**：能脱口讲清 multi-agent 架构决策三件事——什么时候该拆 agent / handoff 怎么设计 / 怎么 eval；并有一个 single-agent vs multi-agent 实验数字作为简历版佐证。

> 注意：上层 chatbot/RAG 不是 scope。课程 C 只关注"agent 架构本身"。

### C-M0 · Agent 基础与工具使用

**Topic**：用最小化代码（≤ 200 行 raw API 循环、不用框架）实现 ReAct / tool use 的本质——model 输出结构化 tool call → host 执行 → 结果回写、循环直到 stop。理解 single-turn function call 与 multi-step agent loop 的区别、tool 失败如何处理、上下文如何管理。这是 C-M1 multi-agent 架构决策的前提。

0. **环境准备**

   **快速 setup**：
   - 实例：`t3.medium`（CPU，Agent 不吃 GPU，~$0.04/hr，几乎免费）；C 课程成本全在 API token 不在机器
   - 装包：`pip install anthropic` 或 `pip install openai`（或用 AWS Bedrock 走 boto3）
   - API key：`.env` + `python-dotenv`；**绝不写进代码**（合规：API key 是凭据，按 production safety rule）
   - 仓库布局：`mkdir -p course-c/m0-mini-agent`
   - 启动：`python mini_agent.py`

   **值得理解**：
   - **tool use loop 的核心循环**：(1) 把 tools schema 传进 API，(2) 拿到 model response，(3) 如果 `stop_reason == "tool_use"` → 执行 tool → 把结果作为 user turn 加回 messages → 回 (2)，(4) 如果 `stop_reason == "end_turn"` → 返回最终答案。**这就是 ReAct 的实现，不要被框架(LangGraph 等)抽象掩盖**
   - **不要用框架做 mini_agent**：直接写 raw API 循环 ≤ 200 行，对理解 tool use 的本质最有帮助。框架（LangChain / LangGraph）留到看完原理再选用
   - **trace logger 是面试可讲的内容**：每一步 (tool, args, result, model_thought) 都落 jsonl；C-M2 eval 阶段会复用这套 log 做 trajectory 评估
   - **tool 失败处理**：把 exception stringify 后作为 tool result 回给 model——让它自己决定重试 / 换工具 / 放弃。**不要在 host 侧 try/except 然后吞掉错误**

1. **学习目标**
   - 解释 ReAct / tool use / function calling 的本质（model 输出结构化 tool call → host 执行 → 结果回写）
   - 区分 single-turn function call 与 multi-step agent loop
   - 理解上下文窗口管理：trajectory 压缩、relevant context selection

2. **学习材料（canonical）**
   - "ReAct: Synergizing Reasoning and Acting in Language Models": https://arxiv.org/abs/2210.03629
   - Anthropic "Building effective agents": https://www.anthropic.com/research/building-effective-agents
   - Anthropic tool use docs: https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview
   - LangGraph concepts (用作概念读物，不强求采用): https://langchain-ai.github.io/langgraph/concepts/

   > 任务相关辅助阅读（API key 安全管理、Anthropic SDK 用法、jsonl trace 格式）放 DOING.md。

3. **Doing 任务** → 见 [`course-c/m0-mini-agent/DOING.md`](course-c/m0-mini-agent/DOING.md)
   - 两个任务：`mini_agent.py`（≤ 200 行 raw API 循环 + 3 工具） / trace logger
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-c/m0-mini-agent/QUIZ.md`](course-c/m0-mini-agent/QUIZ.md)
   - 任务级 mini_agent 完成清单 + trace 输出在 DOING.md
   - QUIZ.md 是 ReAct vs CoT / tool failure handling / context window 管理的概念自查

---

### C-M1 · Multi-agent 架构决策（核心）

**Topic**：Multi-agent 拆分的判定标准 + Handoff 设计（结构化 schema vs 自然语言、共享 memory vs 显式 payload、回环检测）+ 错误处理升级路径。这一层是面试 multi-agent 架构决策的核心；这个模块**只写架构决策文档**，不写新代码——载体是一个 multi-agent 项目（API 文档 → DAG 生成）。

0. **环境准备**

   **快速 setup**：
   - 一个 multi-agent 项目（API 文档 → DAG 生成）本身就是环境——这个模块不写新代码，只写**架构决策文档**
   - 装包：无（只写 markdown）
   - 仓库布局：在项目内 `docs/architecture/` 下放 `agent_topology.md` / `handoff_scenarios.md` / `split_decisions.md`

   **值得理解**：
   - **Handoff 必须用结构化 schema**：定义 `pydantic.BaseModel`（如 `HandoffPayload(task_id: str, parsed_endpoints: list[Endpoint], next_step: Literal["validate", "generate_dag"])`），不要传裸字符串。**面试拷问点**
   - **共享 memory vs 显式 payload**：共享 memory（如 vector store / 全局 state）耦合高、debug 难；显式 payload 强制每次 handoff 把所有需要的 context 显式传——**默认选显式 payload**，除非 payload 大到不合理
   - **回环检测**：每次 handoff 记 `(from_agent, to_agent)`，连续出现 3 次相同对就触发 escalation 到上层 supervisor。简单但有效
   - **拆 agent 的"5 项判定标准"**：(1) 上下文超 100k 时就该拆；(2) 工具集完全不重叠（如读 doc vs 写代码）就该拆；(3) 需要不同 system prompt 风格就该拆；(4) 需要独立 eval 就该拆；(5) **没必要拆就不拆**——multi-agent 是成本（latency、token、debug 复杂度）

1. **学习目标**
   - 拆 agent 的判定标准：上下文污染、工具集差异、专业 prompt、可独立 eval
   - Handoff 设计：消息形式（结构化 vs 自然语言）、状态传递（共享 memory vs 显式 payload）、回环检测
   - 错误处理：哪一层重试、什么情况升级到上层 agent

2. **学习材料（canonical）**
   - Anthropic "How we built our multi-agent research system": https://www.anthropic.com/engineering/built-multi-agent-research-system
   - "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation": https://arxiv.org/abs/2308.08155
   - OpenAI Swarm handoff 设计（参考代码）: https://github.com/openai/swarm
   - "Agent-as-a-Judge: Evaluate Agents with Agents": https://arxiv.org/abs/2410.10934

   > 任务相关辅助阅读（Pydantic schema 规范、回环检测算法、agent supervisor 模式）放 DOING.md。

3. **Doing 任务** → 见 [`course-c/m1-architecture/DOING.md`](course-c/m1-architecture/DOING.md)
   - 三个文档：`agent_topology.md` / `handoff_scenarios.md` / `split_decisions.md`
   - 都在该 multi-agent 项目（API 文档 → DAG）内的 `docs/architecture/` 写
   - DOING.md 内含每个文档的具体结构、必填项、code review checkpoint

4. **过关标准** → 见 [`course-c/m1-architecture/QUIZ.md`](course-c/m1-architecture/QUIZ.md)
   - 任务级 deliverable（三份文档）+ code review checkpoint（handoff 必须用 schema）在 DOING.md
   - QUIZ.md 是拆 agent 5 项判定标准 + handoff 设计 + 错误升级路径的概念自查

---

### C-M2 · Agent eval 体系

**Topic**：Agent eval 三大类——end-to-end（任务成功率） vs trajectory（每步是否合理） vs offline/online/shadow；LLM-as-judge 的偏置（position / length / self-preference）和反偏置手段。这是面试 multi-agent 三件事的最后一件——"怎么 eval"。

0. **环境准备**

   **快速 setup**：
   - 实例：`t3.medium`（CPU 够）；eval harness 跑 LLM-as-judge 的成本在 API token，不在机器
   - 装包：`pip install pytest evaluate scikit-learn`（IoU 等指标）
   - 数据：自己人工标 20 条 (API doc snippet, expected DAG) 样例存 `eval_dataset.jsonl`
   - 仓库布局：在项目 `eval/` 下放 `harness.py` / `eval_dataset.jsonl` / `eval_v0.md`
   - 启动：`pytest eval/test_harness.py` 或 `python eval/harness.py`

   **值得理解**：
   - **end-to-end vs trajectory 不能互相替代**：end-to-end 高但 trajectory 低 = "蒙对了"，下次容易翻车；trajectory 高但 end-to-end 低 = 决策合理但能力不够
   - **DAG 结构匹配**：先把生成的 DAG 和期望 DAG 都规范化（节点 id 排序、忽略无关属性），再算节点集合 IoU + 边集合 IoU 的加权平均
   - **LLM-as-judge 防偏置**：(a) **position bias** → 同一 pair 跑两次交换顺序、取一致；(b) **length bias** → judge prompt 里明确 "ignore length, focus on correctness"；(c) **self-preference** → judge 用 Claude 时被评估的不要也用 Claude（用不同家 model）
   - **20 条够不够**：能发现"完全跑不通"的问题但抓不住 long-tail。production 部署前扩到 ≥ 200 条；面试讲故事 20 条够，但要能说"我意识到这是 sample size 风险"

1. **学习目标**
   - end-to-end eval（任务成功率） vs trajectory eval（每一步是否合理）
   - LLM-as-judge 的常见坑：position bias、length bias、self-preference
   - 区分 offline eval、online eval、shadow eval

2. **学习材料（canonical）**
   - "MLR-Copilot / SWE-bench: Can Language Models Resolve Real-world Github Issues?": https://arxiv.org/abs/2310.06770
   - "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena": https://arxiv.org/abs/2306.05685
   - Anthropic eval guide: https://docs.claude.com/en/docs/test-and-evaluate/eval-tool
   - "AgentBench: Evaluating LLMs as Agents": https://arxiv.org/abs/2308.03688

   > 任务相关辅助阅读（IoU / F1 计算、DAG 同构、judge prompt 模板、人工抽样规模）放 DOING.md。

3. **Doing 任务** → 见 [`course-c/m2-eval/DOING.md`](course-c/m2-eval/DOING.md)
   - 三个任务：标 ≥ 20 条 eval set / 写 end-to-end harness（DAG IoU）+ trajectory harness（LLM-as-judge）/ 跑 baseline 写 `eval_v0.md`
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-c/m2-eval/QUIZ.md`](course-c/m2-eval/QUIZ.md)
   - 任务级 baseline 数字（end-to-end 成功率 + trajectory 合理率）+ harness 代码在 DOING.md
   - QUIZ.md 是 e2e vs trajectory eval / LLM-as-judge 偏置 / sample size 风险的概念自查

---

### C-M3 · Single-agent vs Multi-agent 对照实验（简历版加菜）

**Topic**：在公开 benchmark（SWE-bench Lite 子集 / HumanEval / 自定义任务）上严格控制变量跑 single-agent vs multi-agent，给出 (cost, latency, success rate) 三元组。这是简历版加菜——让你能在面试讲"我做过实验，数据告诉我 multi-agent 不是银弹，X 类型任务才该拆"。

0. **环境准备**

   **快速 setup**：
   - 实例：`t3.medium`（CPU；SWE-bench 跑 docker 测试需要 `t3.large`/`t3.xlarge` 内存大点）
   - 装包：`pip install anthropic swebench`（如选 SWE-bench Lite）
   - 数据：SWE-bench Lite 子集 https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite（300 个 task；选 30–50 个跑 demo）
   - 💰 **主要成本是 API token 不是机器**：~$50/轮（Claude Sonnet × 30 task × ~3 万 tokens）。这是 C 课程唯一显著花钱处
   - 仓库布局：单独的 public repo `multi-agent-vs-single`（这是简历版，公开发）
   - 启动：`python single_agent_baseline.py --task swebench-lite-30`，`python multi_agent.py --task swebench-lite-30`

   **值得理解**：
   - **任务选择**：SWE-bench 是金标准但执行环境复杂（要起 docker 跑测试）；如果嫌重，选 HumanEval / MBPP / 自定义 codegen 任务也行——核心是**任务足够开放、可量化判分**
   - **公平对比**：同 model、同 tools、同输入、同 eval；唯一变量是"single vs multi"
   - **三元组对比**（cost, latency, success rate）缺一不可——只比成功率会让 multi-agent 看起来纯赢，但忽略它贵 3–5×、慢 2–3×
   - **结果可能反直觉**：很多任务 single-agent 配好工具就够了，multi-agent 反而因 handoff 损耗更慢更差。**这是好结论**——面试讲"我做了实验，发现 multi-agent 不是银弹，X 类型任务才该拆"
   - **避免 eval 泄漏**：如果用公开 benchmark，确认 model 训练数据可能包含——这是写 README 必须 disclaimer 的事情

1. **学习目标**
   - 在同一任务上严格控制变量，量化 multi-agent 是否真的赢
   - 写出"我做过实验，数据告诉我 X"的可讲故事

2. **学习材料（canonical）**
   - 复用 C-M1/C-M2 的指定材料即可
   - "Don't Make Your LLM an Evaluation Benchmark Cheater" (避免 eval 泄漏): https://arxiv.org/abs/2311.01964

   > 任务相关辅助阅读（SWE-bench Lite 启动、HumanEval 格式、Anthropic API cost 估算）放 DOING.md。

3. **Doing 任务** → 见 [`course-c/m3-comparison/DOING.md`](course-c/m3-comparison/DOING.md)
   - 三个任务：选 task 准备数据 / `single_agent_baseline.py` + `multi_agent.py` / `experiment_report.md`
   - DOING.md 内含每个任务的 sub-task 拆分、成功/失败标志、辅助阅读、deliverable

4. **过关标准** → 见 [`course-c/m3-comparison/QUIZ.md`](course-c/m3-comparison/QUIZ.md)
   - 任务级 (cost, latency, success rate) 三元组对比表 + 报告在 DOING.md
   - QUIZ.md 是公平对比的控制变量 / multi-agent 输了的归因 / cost-benefit 决策的概念自查

---

### C-M4（辅助）· Slack agent 工具设计提及

**Topic**：工具设计的取舍——粗工具 vs 细工具粒度、schema description 对 model 调用准确率的影响、Read-only vs Write 工具的安全边界、限速 / quota 处理。这是辅助模块——只写文档，不实现 agent，但有面试可讲的"工具设计"细节。

0. **环境准备**

   **快速 setup**：
   - 这个模块**只写文档**（`slack_agent_tools.md`），不实现 agent
   - 仓库布局：在一个 Slack agent 项目内 `docs/tools.md`，或单独写一份外发版本

   **值得理解**：
   - **工具粒度的判断**：粗工具省 round trip 但参数空间大、model 容易调错；细工具调用次数多但每次决策简单。**经验法则**：高频常用动作粗一些，低频危险动作细一些（强迫 model 多步确认）
   - **schema 描述**对 model 调用准确率影响大：参数加 `description` + 给 1–2 个 example 调用、能减少 50%+ 的参数错误
   - **Read vs Write 边界**：所有 write 工具（create_ticket / send_message）必须有 dry-run mode + 调用前 confirmation；这是合规要求也是 production safety
   - **限速 / quota**：Slack API 有严格 rate limit，工具内部要做指数退避；不要让 model 看到 429 错误（让它误以为该换工具）

1. **学习目标**
   - 工具粒度设计：粗工具（"search Slack"）vs 细工具（"list channels" + "get messages"）
   - 工具描述与 schema 对 model 调用准确率的影响
   - Read-only 工具与 write 工具的安全边界（创建 ticket 类操作）

2. **学习材料（canonical）**
   - Anthropic tool use best practices: https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview
   - "Gorilla: Large Language Model Connected with Massive APIs": https://arxiv.org/abs/2305.15334

   > 任务相关辅助阅读（Slack API rate limit、IAM 最小权限、dry-run mode 设计）放 DOING.md。

3. **Doing 任务** → 见 [`course-c/m4-slack-tools/DOING.md`](course-c/m4-slack-tools/DOING.md)
   - 一个任务：`slack_agent_tools.md` 工具表 + 安全设计文档
   - DOING.md 内含具体结构、必填项、code review checkpoint

4. **过关标准** → 见 [`course-c/m4-slack-tools/QUIZ.md`](course-c/m4-slack-tools/QUIZ.md)
   - 任务级 deliverable 在 DOING.md
   - QUIZ.md 是工具粒度 / schema 设计 / read vs write 边界 / 限速处理的概念自查

---

## 对齐核对表

| 成功标准（来自 requirements.md） | 覆盖模块 | 备注 |
|---|---|---|
| 多 GPU 训练：DDP / FSDP / ZeRO 的取舍 | A-M2, A-M3, A-M4 | M2 给出 DDP 实测，M3 渐进升 FSDP 并对比 ZeRO 概念，M4 在多卡上跑通 |
| 通信模式（all-reduce / reduce-scatter / all-gather） | A-M2, A-M3, B-M3 | A-M2 单独 benchmark；A-M3 在 FSDP 中讲 forward all-gather + backward reduce-scatter；B-M3 推理侧 TP all-reduce |
| 显存占用计算 | A-M3 | mem_calc 脚本 + 实测验证 |
| gradient accumulation 与 micro-batch | A-M2 | 含 `no_sync` 用法与 step time 对比 |
| KV cache 设计 | A-M1, B-M1, B-M2 | A-M1 直觉，B-M1 连续 KV cache 实现，B-M2 paged 实现 |
| 为什么需要 PagedAttention | B-M2 | 含 fragmentation / over-provisioning 解释 + 自实现 + 显存对比 |
| Continuous batching 的调度策略 | B-M1 | 自写 scheduler + 与 static batching 对比 |
| Prefill vs decode 的资源差异 | B-M0, B-M1 | B-M0 baseline 给出 TTFT/TPOT 区别，B-M1 调度区分两类请求 |
| Multi-agent 架构决策：什么时候该拆 agent | C-M1 | 拆分判定标准 + 该项目决策记录 |
| Multi-agent 架构决策：handoff 怎么设计 | C-M1 | 5 个 handoff 场景 + 结构化 schema 评审 |
| Multi-agent 架构决策：怎么 eval | C-M2, C-M3 | C-M2 搭 harness，C-M3 single vs multi 对照实验 |

---

**Out-of-Scope 自查**：本课程未把"chatbot/RAG 上层应用、深度数学推导、大段博客或文档撰写、死磕收敛/调参"作为任何模块的学习目标——所有阶段的"过关"都通过口试题、code review checkpoint 或具体 benchmark 数字完成，不要求达到特定 perplexity / 模型 SOTA，也不要求产出长文档。
