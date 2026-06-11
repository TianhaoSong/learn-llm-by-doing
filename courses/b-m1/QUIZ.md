# B-M1 · 知识自查口试

> 验证 KV cache 数据结构 / 调度状态机 / continuous batching 设计是否吃透。30 秒内答得出 = 过。

---

## KV Cache 数据结构（对应任务 1.1）

1. **KV cache 的完整 shape 是什么？为什么 `2` 那一维？**
   - 想点：`[n_layer, 2, max_batch, n_head, max_seq_len, head_dim]`；2 是 K 和 V 各一份；写入时 K 进 [layer, 0, ...] V 进 [layer, 1, ...]

2. **为什么必须预分配整个 max_seq_len，不能 append 时再 grow？**
   - 想点：`torch.cat` 触发新 alloc + memcpy（O(N) 每步、累计 O(N²)）；CUDA allocator 还可能 fragmentation；预分配一次后只在 seq 维写入是 O(1) 每步

3. **为什么 KV cache 的 `max_batch` 维度不能是动态的？**
   - 想点：动态 = 每来新 request 重新 alloc 整个 cache；预分配 max_batch slot + 用 slot id 复用是工业做法；满了就拒绝/排队/preempt

4. **GPT-2 small (n_layer=12, n_head=12, head_dim=64) bf16 max_batch=32 max_seq=2048 的 KV cache 占多少 GB？**
   - 想点：12 × 2 × 32 × 12 × 2048 × 64 × 2 字节 ≈ 2.3 GB；模型本身才 ~250 MB——KV cache 是显存大头

---

## 调度器状态机（对应任务 1.2）

5. **Continuous batching 调度器每 iteration 做哪几件事？**
   - 想点：(1) 释放 finished request 的 KV slot；(2) 从 waiting_queue 拉新 request、allocate slot、加入 prefill；(3) 已在 running 的加入 decode；(4) 跑 forward；(5) 更新 token、检查 max_tokens / EOS

6. **每个 request 有哪些状态？转换条件是什么？**
   - 想点：WAITING（刚 add_request）→ PREFILL（拿到 slot、本 iteration 跑 prefill）→ DECODE（每 iteration 生成 1 token）→ FINISHED（达到 max_tokens 或采到 EOS）

7. **prefill 和 decode 为什么不能简单 concat 进同一 batch？**
   - 想点：attention mask 不同——prefill 是 lower triangular（每个位置只看自己 + 之前），decode 是单 token 看整个 cache；compute pattern 不同（prefill QKV 都是新的 [T, head_dim]，decode Q 是 [1, head_dim] K/V 是 [T, head_dim]）；强行 concat 要写变长 attention（FlashAttention `cu_seqlens`）

8. **如果一个长 prompt 进来需要 prefill 2048 token，会不会卡住正在 decode 的其他请求？怎么缓解？**
   - 想点：会卡住——一个 step 内 prefill 2048 token 占 GPU 几十 ms 期间 decode 都在等；解法 = chunked prefill（把 2048 切成 256 块，每 step 跑一块 + 同时跑 decode），vLLM 的进阶 feature

---

## Continuous Batching 收益（对应任务 2-3）

9. **Continuous batching 比 static batching 主要解了什么问题？两个核心 wins？**
   - 想点：(a) **Padding waste**：static 把短 prompt pad 到最长——continuous 不需要齐头并进；(b) **Head-of-line blocking**：static 里 max_tokens=512 的 request 卡住其他 max_tokens=32 的——continuous 完成的立刻释放

10. **Continuous batching 的 scheduling overhead 是什么？什么场景反而比 static 慢？**
    - 想点：每 iteration 重排队列 + 检查 slot 状态（O(N) 每 step）；如果 batch 大、模型小（每 step 几 ms）、scheduling 占比就大；继续放大 batch / 切大模型反而拉开差距

11. **如果你的 continuous batching 实现只比 static 快 1.2×（远低于 2×），最可能的原因是什么？**
    - 想点：(a) workload 长度方差小（padding waste 不明显）；(b) 模型太小（scheduling overhead 占比大）；(c) prefill 没和 decode 错峰、GPU 资源没复用

---

## 实现细节（覆盖任务 1）

12. **Slot id 和 request id 区别是什么？为什么要分开？**
    - 想点：slot id 是 KV cache 的物理位置（0 到 max_batch-1）、可复用；request id 是请求标识、不可复用；request 完成 free slot 后、slot 被新 request 占用——request id 不能复用

13. **`get_kv(slot_id, layer, end_position)` 切片的语义是什么？为什么需要 `end_position`？**
    - 想点：返回 `[2, n_head, end_position, head_dim]`——只到目前 token 数（不到 max_seq_len，未写入位置是垃圾）；attention 算 `Q @ K.T` 只能看已写入的 KV、不能让未来位置参与

14. **如果 KV cache 写入用 `kv_cache[layer, 0, slot_id, :, position, :] = new_k`，这一行触发新内存分配吗？**
    - 想点：不会——in-place 写入；预分配的好处之一就是这种 indexed assignment 是 O(1) 不动 allocator

---

## 自查标准

- 14 题里 ≥ 12 题 30 秒内答得出 → B-M1 过关
- 题 5-8（调度器状态机）和 题 9-11（CB 收益）是面试核心
