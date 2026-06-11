# B-M3 · 知识自查口试

> 验证 column/row parallel 数学 / 通信拓扑 / 训练 vs 推理并行取舍是否吃透。30 秒内答得出 = 过。

---

## Column / Row Parallel 数学（对应任务 1）

1. **Column-parallel matmul 把 W 沿哪个维度切？输出是 partial 还是完整？**
   - 想点：W = `[in, out]`，沿 out 维（列）切；每 rank 拿 `[in, out/N]` 的一片；输出 `Y_i = X @ W_i` 是 `[B, T, out/N]`、每 rank 持有部分列、不完整（要等 row-parallel 处理）

2. **Row-parallel matmul 把 W 沿哪个维度切？为什么需要 all-reduce？**
   - 想点：W = `[in, out]`，沿 in 维（行）切；每 rank 拿 `[in/N, out]`；同时输入 X 也按 in 维切（接 column-parallel 输出）；`Y_i = X_i @ W_i` 是 `[B, T, out]` 的 partial sum；要 all-reduce 把 N 个 partial sum 加起来才是完整 Y

3. **每个 transformer layer 一共有几次 all-reduce？发生在哪两步？**
   - 想点：2 次；attention output projection（row-parallel）一次 + FFN 第二个 linear（row-parallel）一次

4. **为什么 column-parallel 输出**正好**是 row-parallel 输入？这有什么 trick？**
   - 想点：column-parallel 输出 shape `[B, T, hidden/N]`、各 rank 持有不同列；row-parallel 输入也是 `[B, T, hidden/N]`、各 rank 期待自己那部分行；shape 对齐——所以中间不需要任何通信、直接 chain；这是 Megatron 的核心 idea

5. **QKV 为什么合并成单一大矩阵切？三个独立 Linear 也能切——差别在哪？**
   - 想点：合并 = 一次 matmul（GPU 算力效率高）+ 一次 split heads；分开 = 三次 matmul（kernel launch 开销 3 倍）；shape 对齐结果一样；性能合并版赢

---

## Attention TP（对应任务 1.3）

6. **Attention 为什么沿 head 维度切？切成一半 head 之后中间需要 all-reduce 吗？**
   - 想点：每个 head 独立算 `Q_h @ K_h.T → softmax → @ V_h`、不依赖其他 head；所以 attention 主体（QKV → softmax → @ V）每 rank 独立、零通信；只在 output projection 时合并所有 head 输出 → 那时才 all-reduce

7. **`n_head` 必须能被 `tp_size` 整除——为什么？如果 n_head=12, tp_size=8 行不行？**
   - 想点：每 rank 拿 `n_head / tp_size` 个 head；不整除就有 rank 拿不到 head；12/8=1.5 不行；TP 的 N 选择受 n_head 约束（Llama-2-7B n_head=32, 选 1/2/4/8/16/32 都行）

---

## 通信拓扑（对应任务 3）

8. **NVLink / EFA / TCP 带宽差多少？TP 跨机为什么慢得多？**
   - 想点：NVLink ~600 GB/s（同节点内 GPU 间）；EFA ~100 GB/s（跨节点）；TCP ~10 GB/s（fallback）；跨节点慢一个数量级——TP 每层 all-reduce 在关键路径，慢一个数量级 → 整体慢一个数量级

9. **EFA 配置任何一项错了会退化到 TCP——必查的 4 项是什么？**
   - 想点：(1) cluster placement group；(2) AMI 含 EFA driver；(3) IAM allow EFA；(4) `NCCL_PROTO=Simple` + `FI_PROVIDER=efa`；任一项错 → fallback TCP

10. **跨机 TP=16 vs 单机 TP=8——哪个更快？什么时候必须跨机？**
    - 想点：单机 TP=8 通常更快（NVLink）；必须跨机：单机装不下模型（70B+ 模型显存超 8×80GB）或要更大 batch 提 throughput；跨机 TP 是不得已

---

## 训练 FSDP vs 推理 TP（对应整个 B-M3）

11. **为什么训练用 FSDP 而推理更常用 TP？**
    - 想点：(a) 训练显存大头是 optimizer state（FSDP 切优化器、TP 不切），训练 TP 要配合 sequence parallel 才彻底；(b) 推理无 optimizer state、瓶颈是 KV cache + 吞吐，TP 切 KV cache + 切 weight 通信少；(c) FSDP forward all-gather + backward all-reduce 都很贵、推理无 backward 不需要这套

12. **如果在推理时也用 FSDP（FULL_SHARD），会发生什么？**
    - 想点：每层 forward 进入前要 all-gather params 算完释放——比 TP 更多通信（TP 只在 attention output 和 FFN output 各一次 all-reduce）；FSDP 推理实际可行但慢、TP 是更好选择

13. **为什么 inference batch=1 时 TP 收益小？**
    - 想点：TP 把 compute 切成 N 份、但每层多 2 次 all-reduce；batch=1 单次 forward 时间已经短、all-reduce 的固定开销占比大、scaling efficiency 差；batch 大时 compute 时间长 + all-reduce 摊平更划算

---

## 串联

14. **你的 7B Llama TP=8 NVLink 跑 TTFT_p50 = 50ms、TP=1 = 200ms——scaling efficiency 多少？解释一下。**
    - 想点：speedup = 200/50 = 4×；efficiency = 4/8 = 50%；通信占 50% 时间；这在 7B+NVLink 算合理（13B+TP=8 通常 70%+ efficiency）

15. **如果切到 70B 模型（vLLM 的常见 production size），TP 数应该怎么选？**
    - 想点：70B bf16 = 140GB；单卡装不下（A100 80GB）；TP=2 80GB×2 装得下但跨卡通信开销；TP=4 / TP=8 配 NVLink 是 sweet spot；TP=16 跨机不如 TP=8 + 多副本（pipeline parallel + replica）

---

## 自查标准

- 15 题里 ≥ 12 题 30 秒内答得出 → B-M3 过关
- 题 1-7（TP 数学）和 题 11-13（训练 vs 推理对比）必须答好——这是面试核心
