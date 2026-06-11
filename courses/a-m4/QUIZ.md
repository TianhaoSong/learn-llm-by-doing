# A-M4 · 知识自查口试

> 验证 sharded checkpoint / spot 处理 / MFU 计算 / bottleneck 归因是否吃透。30 秒内答得出 = 过。

---

## 数据 Pipeline（对应任务 2）

1. **为什么大规模训练必须 tokenize 落盘，不能 on-the-fly？**
   - 想点：on-the-fly 把 CPU tokenize 加进 dataloader 关键路径；CPU 的 tokenize 速度远跟不上 GPU 消费 token 的速度（GPU 每秒能算几百 K token）；落盘后 dataloader 只做 mmap 读 + index，飞快

2. **`np.memmap` 比 `np.load` 好在哪？**
   - 想点：mmap 让 OS 按需 page in，启动瞬间不占内存；2 GB 文件 mmap 后 Python 进程内存只多几 MB，访问哪段才 page in 哪段；几十 GB 文件也能直接用

3. **MFU（Model FLOPs Utilization）怎么算？目标值是多少？**
   - 想点：MFU = actual_FLOPs / theoretical_peak_FLOPs；actual ≈ `6 × n_params × tokens_per_step / step_time`（前向 2N + 反向 4N，加 ckpt 是 8N）；A100 bf16 peak = 312 TFLOPS；目标 30-50%；< 20% 是有问题

---

## Checkpoint（对应任务 3）

4. **Sharded vs Full state dict 各什么时候用？**
   - 想点：sharded：训练中 save/load，每 rank 写自己一份，并行写、resume 必须同 N_GPU；full：最终 export，rank 0 gather 所有 shard 写成完整文件，可任意硬件 load

5. **从 ckpt resume 后第一步 loss 跳变（明显高于 save 时）——可能是哪几个原因？**
   - 想点：(1) optimizer state 没 load（AdamW m/v 重置 0）；(2) lr scheduler step 没 resume；(3) RNG 状态没 resume，augmentation 不一致；(4) DistributedSampler 的 epoch 没对上

6. **AWS spot 中断信号是什么？给多少时间？怎么处理？**
   - 想点：SIGTERM；2 分钟；catch 信号 → 触发 sharded ckpt save → exit；下次启动从最新 ckpt resume

7. **FSDP sharded checkpoint 写到 S3 vs 本地 NVMe，各什么场景？**
   - 想点：S3 = 跨实例 resume（spot 中断后换台机器）、durability；本地 NVMe = 同一实例内 resume，写得更快但实例挂了 ckpt 也丢；推荐：步频高的中间 ckpt 写本地 + 偶尔 sync 到 S3

---

## Bottleneck 归因（对应任务 4）

8. **训练 timeline 里的 4 类时间是什么？哪类是 GPU idle？**
   - 想点：(1) compute = forward+backward kernel；(2) communication = NCCL ops；(3) dataloader stall = GPU 等数据，timeline 上是空白；(4) other = optimizer step / H2D copy；(3) 是 GPU idle，要 0 化

9. **如果 profiler trace 显示 NCCL 占 60%，怎么排查？**
   - 想点：(1) bucket 太小？调大；(2) bucket 没 overlap？看 NCCL 是否和 backward 在不同 stream；(3) 互联本来就慢（PCIe / 跨机）？升级硬件；(4) 模型太小、backward 来不及 overlap？放大 micro-batch

10. **如果模型再大 4 倍（350M → 1.4B），瓶颈最可能变成什么？**
    - 想点：(a) Activation 翻 4 倍 → 本来不需要 ckpt 现在需要 → throughput 降 25%；(b) FSDP all-gather buffer 翻 4 倍但 NVLink 带宽不变 → 通信占比上升；(c) param all-gather 时间 ~ 线性增长，overlap 更难 → MFU 下降；具体看是 compute-bound 还是 comm-bound

---

## 串联（覆盖 A 课程整体）

11. **你训了 1B tokens 但 val loss 还在缓慢下降，说明什么？应该继续训还是停？**
    - 想点：还能学 → Chinchilla scaling law 大概 20 tokens/param 是 compute-optimal；350M 模型应该至少 7B tokens 才"训够"；项目目标不是 SOTA、loss 收敛趋于平稳就停（requirements.md 决策原则）

12. **从 A-M0 的 MNIST MLP 到 A-M4 的 1B FineWeb 训练——训练 loop 本质有哪些没变？**
    - 想点：(1) `for batch: zero_grad → forward → loss → backward → step` 这个 5 行循环没变；(2) `train()/eval()` 切换没变；(3) data fetch + H2D copy 没变；变了的是模型大小、并行策略、precision、dataloader 复杂度

13. **如果面试官问"你这个项目最大的工程挑战是什么"——你怎么用 A-M4 的经历回答？**
    - 想点：可选角度——(a) FSDP `auto_wrap_policy` 配错通信极慢调试；(b) sharded ckpt 的 N_GPU 不一致 resume 问题；(c) MFU 从 12% → 35% 的归因过程；(d) spot 中断 + resume 的 idempotency；选一个有具体数字 + trace 截图能讲的

---

## 自查标准

- 13 题里 ≥ 11 题 30 秒内答得出 → A-M4 过关 → 课程 A 完成
- 题 8-10（bottleneck 归因）必须答好——是面试官最爱挑战的点
