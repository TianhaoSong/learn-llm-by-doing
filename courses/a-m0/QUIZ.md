# A-M0 · 知识自查口试

> 不是面试演练，是验证 A-M0 知识是否吃透。每题脑子里 30 秒内答得出 = 过；答不出回去看 DOING.md 对应任务或学习材料。

---

## Autograd & Training Loop（对应任务 1）

1. **为什么 `loss.backward()` 之后必须 `optimizer.zero_grad()`？不 zero 会发生什么？**
   - 想点：grad 默认 accumulate；不 zero 会让本 step 的 grad 累加上一 step 的，变成隐式 gradient accumulation

2. **`.detach()` / `torch.no_grad()` / `retain_graph=True` 各自什么场景用？**
   - 想点：detach = 截断单个 tensor 的图；no_grad = context manager 关掉一段代码的图构建（eval / inference）；retain_graph = 同一 forward 要 backward 两次（如 GAN、二阶导）

3. **`model.train()` 和 `model.eval()` 切换了哪些行为？MLP 没 dropout/BN 是不是就不需要切？**
   - 想点：dropout、BatchNorm 的 running stats、有些自定义层；MLP 上确实不影响输出，但养成习惯——后面 transformer 一定有 dropout

4. **CrossEntropyLoss 的输入应该是 logits 还是 softmax 后的概率？为什么？**
   - 想点：logits；CE 内部做 log_softmax + nll，自己再 softmax 一次会数值不稳定（log(0)）

---

## DataLoader & CPU→GPU Pipeline（对应任务 2）

5. **DataLoader `num_workers > 0` 时为什么会出现 fork/spawn 问题？为什么 Windows/Mac 容易踩？**
   - 想点：worker 是子进程；Linux 默认 `fork`（共享地址空间），Windows/Mac 默认 `spawn`（重新 import 主模块）——主模块顶层有副作用代码就会重复执行；解决靠 `if __name__ == '__main__':` guard

6. **`pin_memory=True` 干了什么？为什么搭配 `non_blocking=True` 才有用？**
   - 想点：pin = 锁页内存（不可被换出到 swap）；async H2D copy 必须从锁页内存 DMA；非锁页的 copy 是同步的，`non_blocking=True` 也没用

7. **`persistent_workers=True` 解决什么问题？什么场景没必要开？**
   - 想点：避免每个 epoch 重建 worker（fork/spawn + dataset init 都要重来）；epoch 很长 / 很少 epoch 时收益不明显

8. **如果 GPU compute 已经是瓶颈（GPU util 100%），调 dataloader 还有意义吗？**
   - 想点：没意义——dataloader 调优是把"GPU 等数据"变成"GPU 一直算"；GPU 已经满了说明数据 prefetch 已经够了

---

## CUDA 异步执行（对应任务 3）

9. **`tensor.cuda()` 这一行返回时 copy 真的完成了吗？什么时候必须显式 sync？**
   - 想点：默认 blocking copy（同步——CPU 等到 copy 完才返回）；`non_blocking=True` 才是 async（CPU 不等）；sync 必要场景：CPU 要读 GPU tensor 的值（`.item()` / `.cpu()` 自动 sync）、跨 stream 依赖、计时

10. **为什么 `time.time()` 测 GPU 时间是错的？正确方法是什么？**
    - 想点：CPU 看到 kernel launch 完就走了，GPU 还在算；`time.time()` 测的是"launch 到 launch"不是"compute 到 compute"；正确：`torch.cuda.Event` + `event.record()` + `event.synchronize()` + `event.elapsed_time(other)`

11. **同一个 stream 内的 op 是并行还是串行？两个不同 stream 默认呢？**
    - 想点：同 stream 串行；不同 stream 之间默认并行（除非显式 sync 或有数据依赖）

12. **`non_blocking=True` 在什么前提下才真的 async？**
    - 想点：源 tensor 必须在 pin_memory；目标必须在 GPU；DataLoader 配 `pin_memory=True` + 训练循环 `x.to(device, non_blocking=True)` 才是完整 async pipeline

---

## 综合（跨任务）

13. **你的 train_mlp.py 训练 step 的时间分布大概是：dataloader fetch / H2D copy / forward / backward / optimizer step。哪一段最容易成为瓶颈？怎么验证？**
    - 想点：MLP 太小时 dataloader 是瓶颈（任务 2 的发现）；用 `torch.profiler` 抓一段或者手动 `cuda.Event` 卡每段时间

14. **如果同事说"我的训练只用了 GPU 的 30%，怎么办"，你按什么顺序排查？**
    - 想点：(1) GPU util 低 = compute 在等什么 → (2) 看 dataloader（任务 2 那套）→ (3) 看是不是 batch_size 太小 → (4) 看 forward/backward 内部是不是有 CPU↔GPU 同步点（如 print loss.item() 每步）

---

## 自查标准

- 14 题里能 30 秒内答出 ≥ 12 题 → A-M0 过关
- 答不出的题：标记下来，回去重做对应任务 / 重读对应学习材料；一周后再答一遍
- 完全不需要写下答案，能脑内组织出"主干 + 一两个细节"就够
