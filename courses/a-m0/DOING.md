# A-M0 · Doing 任务详细规格

> 这个模块要建立单卡训练的三块基础直觉：模型怎么训（训练循环）、数据怎么喂到 GPU 不让它干等（dataloader）、GPU 异步执行是怎么回事（stream）。三个任务一一对应，做完它们后面加多卡、加优化才有地基。
>
> - `train_mlp.py` —— 训练循环 + autograd
> - `bench_dataloader.py` —— CPU→GPU 数据 pipeline 的性能
> - `cuda_stream_demo.py` —— GPU 异步执行

---

## 任务 1 · `train_mlp.py`

### 为什么做这个
后面所有训练代码——从单卡 GPT 到多卡 1B——骨架都是同一个五步循环：前向算输出、算 loss、反向求梯度、更新参数、清空梯度。这个循环你得能闭着眼写出来，不然后面加 DDP、加混合精度时，会分不清哪些是循环本身、哪些是叠加上去的东西。用最简单的 MNIST 分类把这个循环跑通，是整个项目的地基。

### 目标
不查文档能从零写出"PyTorch 训练循环"——`nn.Module` 子类 → `forward` → `loss.backward()` → `optimizer.step()` → `zero_grad()` → 切 train/eval mode。

### Sub-tasks（推荐顺序）
1. **数据**：用 `torchvision.datasets.MNIST(root='./data', download=True, transform=transforms.ToTensor())` 拿到 train/test 两个 dataset，包成两个 DataLoader（先 `num_workers=0` 跑通；调优在任务 2）
2. **模型**：写一个 `class MLP(nn.Module)`，结构 `784 → 128 → 64 → 10`（ReLU 中间）。`forward` 里第一行 `x.view(x.size(0), -1)` flatten
3. **Loss + optimizer**：`nn.CrossEntropyLoss()` + `torch.optim.AdamW(model.parameters(), lr=1e-3)`
4. **Train loop**：写 `train_one_epoch(model, loader, optimizer, criterion, device)`——里面顺序是 `model.train()` → for batch → `optimizer.zero_grad()` → forward → loss → `loss.backward()` → `optimizer.step()`
5. **Eval loop**：写 `evaluate(model, loader, device)`——`model.eval()` + `with torch.no_grad():`，返回 (loss_avg, acc)
6. **Main**：跑 5 epoch，每 epoch 末打 `(epoch, train_loss, eval_loss, eval_acc)`

### 成功标准
- Eval accuracy ≥ **97%**（MLP 在 MNIST 上 5 epoch 内能稳定到 97%+；到不了大概率 loop 写错了）
- Loss 曲线单调下降趋于平稳（不要求画图，print 出来肉眼看即可）
- Code review checkpoint：
  - `model.train()` 和 `model.eval()` 各出现至少一次
  - `optimizer.zero_grad()` 在 step 循环内出现（不是循环外）
  - eval 路径在 `with torch.no_grad():` 内

### 失败排查
- **Loss 不动（卡在 ~2.3 = ln(10)）**：optimizer.step() 漏了 / lr 数量级不对
- **Loss 下降但 acc 卡在 10%**：eval loop 的 `model.eval()` 漏了，或者 acc 计算用了 logits 没 argmax
- **Loss 第一个 epoch 后还在缓慢下降但 step time 巨慢**：`zero_grad()` 漏了，grad 在 accumulate
- **Train loss << eval loss 且 eval acc 低**：dropout/BN 没切 eval mode（这个 MLP 没 BN，但养成习惯）

### 辅助阅读（非 canonical）
- 60-min Blitz §4 "Training a Classifier" 是直接模板（CIFAR-10 改成 MNIST）

### Deliverable
- `course-a/m0-pytorch/train_mlp.py`
- 文件末尾 docstring 或同目录 `notes.md` 一段话："我踩到的坑 + 怎么发现的"（即使没踩到也写一句"一次跑通"——这一段是给 A-M1 之后回看用的）

---

## 任务 2 · `bench_dataloader.py`

### 为什么做这个
训练慢，第一反应往往是"GPU 不够快"——但很多时候 GPU 是在**干等数据**：CPU 还没把下一批数据读好、传到显卡上。PyTorch 的 DataLoader 有几个旋钮专门治这个（多进程预读、锁页内存、复用进程），但光看文档记不住、也没感觉。这个任务让你把这几个旋钮逐个开关、亲眼量出每种组合快多少——以后写训练代码就会条件反射地把它们设对，而不是让 GPU 空转。

### 目标
亲眼看到 `num_workers / pin_memory / persistent_workers` 三个旋钮对 step time 的影响，建立"DataLoader 性能模型"的直觉。以后写训练代码不会忘记设。

### Sub-tasks
1. **写一个 `bench(num_workers, pin_memory, persistent_workers) -> step_time_ms` 函数**：
   - 构造 DataLoader（用任务 1 同款 MNIST，但 batch_size 提到 256）
   - **bench 函数内部不要跑真模型**——只做 `x.to(device, non_blocking=True)` + 一个轻 op 强制 transfer 完成（如 `_ = x.sum()`）。这样 step time 主要是 fetch + transfer，dataloader 旋钮的差异才看得出来；如果跑真模型，GPU compute 时间会盖住 dataloader 差异、所有组合数字几乎一样
   - 跑 50 个 step，丢弃前 5 步 warmup
   - 计时：CUDA 用 `torch.cuda.Event` + `event.synchronize()`；MPS / CPU 用 `time.perf_counter()`（无 `cuda.Event` API）
   - 返回中位数 step time（用 median 不用 mean——避免单 step 极慢被 OS 调度 / GC 拉偏）
2. **遍历组合**：`itertools.product([0, 2, 4], [False, True], [False, True])` = 12 组
   - `num_workers=0` 时 `persistent_workers` 必须是 False（PyTorch 限制）——这种组合直接跳过
3. **打印结论表**：
   ```
   num_workers=0, pin=F, persist=F: 12.3 ms
   num_workers=0, pin=T, persist=F:  9.8 ms
   ...
   num_workers=4, pin=T, persist=T:  2.1 ms
   ```
4. **结论一句话**：在 README / 文件末尾打一行 "最快组合是 (X, Y, Z)，因为 ..."

### 成功标准
- 至少 8 组组合的数字（去掉无效组合）
- 最慢组合 vs 最快组合 step time 差 ≥ **2×**（差不到说明 GPU compute 是瓶颈、dataloader 没空间——这种情况把 batch_size 调小、或者把模型从 MLP 换成更轻的 forward 来放大 dataloader 占比）
- 一句话结论解释为什么这个组合最快（提到"主进程不等 IO" / "锁页内存让 H2D async 起来" / "worker 复用"）

### 失败排查
- **`num_workers=4` 比 `num_workers=0` 还慢**：可能是 Mac/Windows 的 `fork` vs `spawn` 问题；或者 dataset 本身在内存里（MNIST 全 load 了），worker 启动开销 > 收益。换更大的 dataset（CIFAR-10）或在 Linux 上跑
- **`pin=True` 和 `pin=False` 数字几乎一样**：在 MPS / CPU 上是预期——unified memory 没 H2D copy 概念；CUDA 上才有差异，且必须配 `non_blocking=True` + 实际 GPU compute 才看得出 overlap 效果

### 辅助阅读（非 canonical）
- PyTorch DataLoader docs（看 `num_workers` / `pin_memory` / `persistent_workers` 的描述）：https://pytorch.org/docs/stable/data.html
- PyTorch performance tuning guide §"Enable async data loading"：https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

### Deliverable
- `course-a/m0-pytorch/bench_dataloader.py`
- 同目录 `bench_dataloader_results.md`：表格 + 一段一句话结论

---

## 任务 3 · `cuda_stream_demo.py`

### 为什么做这个
GPU 是异步执行的：你写 `x.cuda()` 那行返回时，数据其实还没传完——CPU 只是把任务丢给 GPU 排队、就接着往下走了。不懂这一点，后面两个坑躲不开：(1) 用 `time.time()` 测 GPU 耗时会量出错的数字（CPU 早走了，没等 GPU 算完）；(2) 看不懂为什么数据传输能和计算"同时"发生。这个任务用两条 stream 让一次数据传输和一次矩阵乘**并行跑**，亲眼看到它们时间重叠——把"GPU 异步"从一句话变成你见过的现象。后面所有 profiling、所有"为什么没加速"的排查，都建立在这个直觉上。

### 目标
亲眼看到 GPU 异步执行——同一个程序里不 sync 时 op 乱序、加 sync 后串行。建立"`tensor.cuda()` 返回时 copy 没完成"的体感，这是后续所有 profiling 工作的前提。

### Sub-tasks
1. **构造两个 stream**：
   ```python
   s1 = torch.cuda.Stream()
   s2 = torch.cuda.Stream()
   ```

2. **准备 3 个 tensor**（角色明确，不要混用）：
   ```python
   # H2D copy 的源——CPU 上、pinned；pin 是 async copy 的前提
   x_cpu = torch.randn(4096, 4096, pin_memory=True)

   # matmul 的两个输入——直接在 GPU 上构造，独立于 x_cpu/x_gpu
   a = torch.randn(8192, 8192, device='cuda')
   b = torch.randn(8192, 8192, device='cuda')
   ```
   **关键**：a/b 必须**独立于** H2D copy 的 tensor。如果你写成 `y = x_gpu @ x_gpu`（matmul 依赖 H2D 的输出），s2 会自动等 s1 完成（数据依赖），就**观察不到并发**——任务 3 就失败了。

3. **场景 A：不 sync**
   ```python
   # 4 个 event 卡住每个 op 的开始 / 结束时间
   copy_start = torch.cuda.Event(enable_timing=True)
   copy_end   = torch.cuda.Event(enable_timing=True)
   mm_start   = torch.cuda.Event(enable_timing=True)
   mm_end     = torch.cuda.Event(enable_timing=True)

   with torch.cuda.stream(s1):
       copy_start.record(s1)
       x_gpu = x_cpu.to('cuda', non_blocking=True)
       copy_end.record(s1)

   with torch.cuda.stream(s2):
       mm_start.record(s2)
       y = a @ b
       mm_end.record(s2)

   torch.cuda.synchronize()  # 必须先 sync 才能读 elapsed_time

   # CUDA event 没有绝对 wall-clock，只能算两 event 间的差值
   # 用 copy_start 当 t=0 锚点，其他都是"距离开始多久"
   t_copy_end = copy_start.elapsed_time(copy_end)   # 返回 ms
   t_mm_start = copy_start.elapsed_time(mm_start)
   t_mm_end   = copy_start.elapsed_time(mm_end)
   print(f"copy:   [  0.00, {t_copy_end:6.2f}] ms")
   print(f"matmul: [{t_mm_start:6.2f}, {t_mm_end:6.2f}] ms")
   print(f"重叠？  {'是' if t_mm_start < t_copy_end else '否'}")
   ```

4. **场景 B：加 sync**——在 H2D 之后立刻 `s1.synchronize()`、再 launch matmul，重复场景 A 的计时

5. **观察现象**：
   - 场景 A：[copy_start, copy_end] 和 [mm_start, mm_end] 时间区间**重叠**（mm_start < copy_end）→ 两个 stream 真的并发
   - 场景 B：完全串行（mm_start ≥ copy_end）

6. **写 README**：贴两组时间戳 + 一段话解释"为什么不 sync 时是并发的"

### 成功标准
- 场景 A 看到时间戳重叠（matmul_start < copy_end）
- 场景 B 看到完全串行（matmul_start ≥ copy_end）
- README 一段话解释，必须提到："不同 stream 之间默认并行" + "同一 stream 内 op 串行" + "`torch.cuda.Event` 是 GPU 端时间，不是 CPU 端"

### 失败排查
- **场景 A 也是串行**：
  - op 太小被合并执行了 → 把 matmul 放大到 8192×8192 或重复 10 次
  - 不小心都丢在 default stream → 显式用 `with torch.cuda.stream(s1):` context manager 包住每段 op
  - H2D copy 没用 `pin_memory` → 锁页内存是 async copy 的前提
- **`event.elapsed_time` 报错**：必须先 `torch.cuda.synchronize()` 让所有 event 都 record 完才能读

### 辅助阅读（非 canonical）
- PyTorch CUDA stream 示例代码（在 CUDA semantics 文档同页）

### Deliverable
- `course-a/m0-pytorch/cuda_stream_demo.py`
- 同目录 `cuda_stream_demo.md`：贴两组 timestamp + 解释段

---

## 三个任务做完之后

- 跑一遍 `QUIZ.md`，自答口试题（不要求录音/写下来——能在脑子里 30 秒内答出来即过）
- 答不出来的题，回去找对应任务的 sub-task 或学习材料
- 全部能答出来 → A-M0 过关，开 A-M1
