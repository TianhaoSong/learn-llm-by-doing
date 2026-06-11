# A-M2 · 知识自查口试

> 验证 NCCL 通信 / DDP 内部机制 / grad accumulation / mixed precision 是否吃透。30 秒内答得出 = 过。

---

## NCCL Collective Ops（对应任务 1）

1. **All-reduce / reduce-scatter / all-gather 三个 op 各自做什么？为什么 DDP 用 all-reduce？**
   - 想点：all-reduce = 所有 rank 上的 tensor 求和（或其他 op）后所有 rank 都拿结果；reduce-scatter = 求和后按 rank 切片每人拿一段；all-gather = 每人有一段、collect 成完整的；DDP 要让所有 rank 同步同一份梯度 → all-reduce 正好对位

2. **Algorithm bandwidth vs bus bandwidth 的差别？为什么 NCCL 报两个数字？**
   - 想点：algo_bw = `tensor_bytes / time`（用户视角传了多少数据）；bus_bw = `algo_bw × correction_factor`（all-reduce 的 factor 是 `2(N-1)/N`，反映 ring 算法每个 link 实际通过的字节数）；用 bus_bw 比较不同 op 的硬件利用率才公平

3. **为什么小 tensor 的 all-reduce 带宽极低？这与 DDP bucket 的设计有什么关系？**
   - 想点：小 tensor 是 latency-bound（每次 op 有固定 launch + handshake 开销）；DDP bucket 把多个小梯度拼成大 tensor 一起 all-reduce，提升带宽利用率

4. **NVLink (~600 GB/s) / PCIe (~32 GB/s) / Ethernet (~10 GB/s) 三种互联，DDP 的 scaling 在哪种上掉得最厉害？**
   - 想点：Ethernet 最差（跨机），PCIe 中（g5），NVLink 最好（p4d/p4de/H100）；scaling efficiency 跟 backward 时间和 all-reduce 时间的比例有关——通信慢就 overlap 不掉

---

## DDP 内部机制（对应任务 2）

5. **DDP 是在 forward 还是 backward 触发 all-reduce？为什么是 backward？**
   - 想点：backward；因为梯度是 backward 算出来的；放在 backward 还能和后续层的 backward overlap

6. **DDP bucket 的作用是什么？bucket size 调大调小各影响什么？**
   - 想点：把多个 param 的梯度拼成一个大 tensor 做 all-reduce（减少 op 次数、提升带宽）；调大 = 通信更高效但 overlap 窗口少（要等 bucket 塞满才发）；调小 = 通信开销大但 overlap 更细

7. **`find_unused_parameters=True` 的开销是什么？什么时候必须开？**
   - 想点：开销 = 每次 backward 后扫一遍参数图找未使用的；常见场景是模型有条件分支（if-else 走不同子图），不开会卡住等不存在的梯度；不需要时绝不开（性能损失）

8. **DDP 同步发生在 backward——但如果某个 rank 的 backward 比别人快，它会发生什么？**
   - 想点：等；每个 bucket 的 all-reduce 都是同步点，快的 rank 在 collective op 里阻塞等慢的；这就是为什么 dataloader stall 在多卡下放大（一个 rank 慢拖累全部）

---

## Gradient Accumulation（对应任务 3）

9. **Global batch = 256，单卡显存只够 micro-batch=8，DDP×4 卡，应该怎么配置 grad accumulation？**
   - 想点：global = micro × N_gpu × accum_steps → 256 = 8 × 4 × 8，所以 accum_steps=8

10. **`model.no_sync()` 是干什么的？不用它会怎样？**
    - 想点：DDP 默认每次 backward 都触发 all-reduce；accum 时前 `accum-1` 次本地累加梯度不需要同步；`no_sync()` 跳过这些 step 的 all-reduce、最后一个 step 才同步；不用 = 每次 micro-batch 都全网 all-reduce、慢 accum 倍

11. **`loss = loss / accum_steps` 这个除法在干什么？不除会怎样？**
    - 想点：因为多次 backward 梯度是累加的；不除 → 等价于 lr 放大了 accum_steps 倍，训练曲线和直接大 batch 不一致

12. **固定 global batch 不变，micro-batch 越大 step time 越快——为什么？代价是什么？**
    - 想点：少了 accum 次数 = 少了 forward/backward 调度开销 + activation 计算更并行；代价是 activation 显存峰值更高

---

## Mixed Precision（对应任务 4）

13. **fp16 vs bf16 的数值范围 / 精度差在哪？**
    - 想点：fp16 = 5 exp + 10 mantissa，范围窄（max ~65504）但精度高；bf16 = 8 exp + 7 mantissa，范围和 fp32 一样（max ~3.4e38）但精度低；训练里 overflow 比精度损失更致命，所以 bf16 更稳

14. **fp16 训练为什么需要 GradScaler？bf16 为什么不需要？**
    - 想点：fp16 梯度小到 underflow 成 0 是常见问题；GradScaler 把 loss 乘一个大常数（128/512/1024）让梯度落进 fp16 表示范围、step 前再除回去；bf16 范围够宽不会 underflow，不需要 scaler

15. **AMP autocast 把哪些 op 跑成 fp16/bf16，哪些保留 fp32？**
    - 想点：matmul / conv / activation 跑半精度（吃带宽和 compute）；reduce / softmax / norm 保留 fp32（数值敏感）；loss 计算保留 fp32

16. **mixed precision 把 activation 显存降了一半，但 optimizer state 还是 fp32——为什么？**
    - 想点：AdamW 的 m / v / param 副本都是 fp32 才稳定（精度敏感）；这是 ZeRO/FSDP 之外另一个能省显存的点（A-M3 会展开）

---

## 综合（跨任务）

17. **你跑出 4 卡 scaling efficiency 只有 50%，怎么排查？**
    - 想点：(1) 看 tokens/sec/GPU 是不是 dataloader 限的（A-M0 任务 2 那套）；(2) profiler trace 看 NCCL 和 backward 是不是真 overlap；(3) 看是不是 small model（backward 太快，all-reduce 来不及 overlap）；(4) 看是不是 PCIe 互联本来就慢

18. **如果你的训练只能用 fp32（比如某些自定义 op 不支持 autocast），DDP scaling 会更差还是更好？**
    - 想点：更好（相对而言）；fp32 forward/backward 时间更长 → 通信 overlap 窗口更大、scaling 更好；这是为什么"高精度训练 + DDP" scaling 看起来反而漂亮的原因，但绝对 throughput 慢

---

## 自查标准

- 18 题里 ≥ 15 题 30 秒内答得出 → A-M2 过关
- 通信原语带宽（题 1-4）和 DDP 内部机制（题 5-8）是面试必考——要 100% 准
