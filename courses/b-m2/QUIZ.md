# B-M2 · 知识自查口试

> 验证 PagedAttention 的 motivation / 数据结构 / 实现取舍是否吃透。30 秒内答得出 = 过。

---

## Motivation（对应任务 2）

1. **B-M1 的连续 KV cache 在变长输出下浪费显存有哪两种来源？**
   - 想点：(a) **Over-provisioning**：每 sequence 按 max_tokens 预算（实际 max_tokens=256 但生成 32 → 87% 浪费）；(b) **Internal fragmentation**：每 sequence 在自己 reserved 的 max_seq_len 区间内、不能被其他 sequence 用

2. **同样 24GB 显存预算，为什么 paged 能并发更多 request？**
   - 想点：paged 按实际生成长度按 block 分配（写满一 block 才申请下一 block），未用的 block 留给别人；连续 cache 必须按 max_tokens 预留全程

3. **PagedAttention 类比的是 OS 哪个机制？为什么 16 token / block 而不是 1 token / "byte" 粒度？**
   - 想点：OS 虚拟内存的 page table；粒度太细（1 token）→ block_table 巨大 + 每次 attention 要 indirect access 太多次；太粗（如 1024 token）→ 又退化成 over-provisioning；16 是经验最优

---

## 数据结构（对应任务 1.1）

4. **`physical_blocks` 是什么 shape？`block_table` 是什么类型？**
   - 想点：`physical_blocks[total_blocks, n_layer, 2, n_head, block_size, head_dim]`；`block_table: dict[seq_id, list[block_id]]`（或同等结构）

5. **`free_blocks` 用 `set` 不用 `list`，为什么？**
   - 想点：allocate / free 都是 O(1)（set 的 pop 和 add）；list 的 remove 是 O(N)、当 block 数几千时 scheduling overhead 显著

6. **prefill 一个新 request 时，block 是怎么分配的？**
   - 想点：算 `num_blocks = ceil(prompt_len / block_size)`；从 free_blocks 弹出对应数量、加进 block_table[seq_id]；不够 → 拒绝/排队（B-M2 简单做法）或 preempt（vLLM 做法）

---

## 实现取舍（对应任务 1.2）

7. **PagedAttention 用 PyTorch + `torch.gather` 实现，单 op 比连续 cache 慢 30-50%——为什么？**
   - 想点：每 step `physical_blocks[block_ids]` 触发 indirect memory access、GPU 端不连续读、效率低；vLLM CUDA kernel 直接在 block 上算（pointer dereference 在 kernel）、不拼回——这才是它快的原因

8. **PagedAttention 的"显存收益"和"单 op 开销"是两件分开的事——具体差别？**
   - 想点：显存收益 = 数据结构层（fragment/over-provision 减少），实现无关；单 op 开销 = kernel 实现层（PyTorch index_select 慢、CUDA kernel 快）；vLLM 用 CUDA kernel = 拿到显存收益 + 抹平开销 → 纯赢；PyTorch 实现 = 只拿显存收益 → 单 op 慢

9. **为什么 vLLM 用 block table 索引、而不是把 KV 直接打散成 per-token？**
   - 想点：per-token 索引粒度太细——block_table 大小翻 16 倍 + 每个 attention step 要 dereference 几千个 token 而不是几百个 block；block 是粒度的甜点

10. **`block_size` 选大了和选小了各有什么代价？**
    - 想点：太小（如 4）= block_table 巨大 + indirect access 频繁 → 性能差；太大（如 64）= 内部 fragmentation 回来了（一个 sequence 长 17 token 也占 64 token block）；vLLM 默认 16 是经验最优

---

## 调度（对应任务 1.3）

11. **OOM 时调度器应该怎么决策？vLLM 怎么处理？**
    - 想点：B-M2 简单做法 = **拒绝新 request、加回 waiting_queue**；vLLM 做法 = **preempt 已有 DECODE 请求**（把 KV swap 到 CPU 或直接释放 + 重新跑 prefill），避免 head-of-line blocking

12. **Copy-on-write 在什么场景用？和 prefix caching 什么关系？**
    - 想点：多个 sequence 共享相同 prompt prefix 时（如 system prompt），多 request 的 block_table 指向同一物理 block；任何一个 request 要写入新 token 时、触发 copy（COW）只复制那一个 block；prefix caching 是这个机制的具体应用 = system prompt 全局共享

---

## 串联

13. **如果你的 paged engine 在 B-M2 里没看到显存收益（max_concurrent 没涨）——最可能的原因是什么？**
    - 想点：(a) workload 的 max_tokens 和实际生成长度差不多（over-provisioning 本来就少）；(b) 实际生成长度都 < block_size、每 seq 一个 block 也是浪费；(c) `block_table` 没回收 finished sequence 的 block

14. **为什么本模块不以 throughput 为过关指标？面试官问"你的 PagedAttention 比连续 cache 慢"怎么答？**
    - 想点：本模块只验证显存收益是否成立；性能收益要 vLLM 的 CUDA kernel 才能拿到；面试答："PyTorch 实现只能拿数据结构层的收益、kernel 层的收益要专门写 CUDA；这是为什么 vLLM 是 production-grade、我的实现是 demo-grade，差距在 B-M4 归因得很清楚"

---

## 自查标准

- 14 题里 ≥ 12 题 30 秒内答得出 → B-M2 过关
- 题 7-10（实现取舍）必须答好——这是 PagedAttention 面试最深入的考点
