# B-M4 · 知识自查口试

> 验证 FlashAttention / CUDA graph / RadixAttention / 量化等高级优化机制是否有概念。30 秒内答得出 = 过。

---

## FlashAttention（对应任务 3 归因）

1. **FlashAttention 解决了什么问题？为什么是 IO-bound 优化而不是 compute-bound？**
   - 想点：标准 attention 算 `Q @ K.T`（[T, T] 矩阵）要写到 HBM、再读出来做 softmax；T 大时 HBM 带宽是瓶颈、不是 compute；FlashAttention 用 tiling 把 Q/K/V 分块、每块在 SRAM（GPU shared memory）算完 softmax + @ V、只写最终结果到 HBM——HBM 读写次数从 O(T²) 降到 O(T)

2. **FlashAttention v2 比 v1 主要改了什么？**
   - 想点：v2 改了 work partition、把 outer loop 从 K/V 切到 Q 切；好处是每个 thread block 输出一段连续 Q output、写 HBM 减少；性能 2× v1

3. **FlashAttention 为什么对 prefill 收益大、对 decode 收益小？**
   - 想点：prefill T 大 → attention 矩阵 [T, T] 大 → HBM IO 是瓶颈 → FA 大赢；decode T=1 → attention 矩阵 [1, N] 小 → HBM IO 没那么糟 → FA 收益小（KV cache 读才是 decode 的瓶颈）

---

## CUDA Graph（对应任务 3 归因）

4. **CUDA graph 解决了什么问题？什么场景收益大？**
   - 想点：每个 PyTorch op launch kernel 有 ~5-20 μs Python+CUDA driver overhead；decode 每 step launch 几百个 kernel → 累计几 ms 是不可忽略 overhead；CUDA graph 把整段 op 序列 capture 一次、之后 replay 几乎零 overhead；对 decode（小 batch + 多 small kernel）收益大

5. **CUDA graph 的限制是什么？为什么 PagedAttention 要专门配合？**
   - 想点：capture 时 input shape / 控制流必须固定；PagedAttention 的 block_table 每次不同——vLLM 实现是把 block_table 做成 static buffer、每次 capture 时填进去（不重 capture）；这是配合点

---

## RadixAttention（对应任务 3 归因 / SGLang）

6. **SGLang 的 RadixAttention 是什么？比 vLLM 的 prefix caching 强在哪？**
   - 想点：用 radix tree 维护已缓存的 KV、共享前缀的 sequence 自动复用；vLLM 也有 prefix caching 但只匹配整 block 的前缀；RadixAttention 粒度更细（per-token 前缀匹配）；对 multi-turn / system prompt 共享重的 workload 收益大

7. **什么 workload 上 RadixAttention 比 vLLM prefix caching 显著？**
   - 想点：(1) 长 system prompt 多 user 共享；(2) ReAct agent loop 反复 prepend 历史；(3) RAG 检索结果共享 chunks；如果 workload 没共享前缀（独立 prompt）→ 两者差不多

---

## 量化（对应任务 3 归因）

8. **fp8 / int4 / AWQ 各是什么？什么时候用？**
   - 想点：fp8 = H100 原生支持的 8-bit float（精度好于 int8、范围接近 fp16）；int4 = 4-bit 整数、显存只剩 1/4 但精度损失明显；AWQ = activation-aware weight quantization、保留对 activation 重要的 weight 精度；量化是显存压力 / 推理速度优化、不是训练手段

9. **推理时量化和 mixed precision 训练有什么区别？**
   - 想点：训练 mixed precision = forward/backward 用半精度、optimizer 保 fp32；推理量化 = weight 一次性转低精度存盘 + activation 算时也低精度（或保 fp16）、不需要训练；推理量化只 inference time 用、训练时模型还是 fp32/bf16

---

## 三方对比（对应任务 2-3）

10. **你的 self-engine 比 vLLM 慢 3×、最大差距来自哪里？**
    - 想点：FlashAttention（self 用 PyTorch SDPA、vLLM 用 FA2）≈ 50% 差距；PagedAttention CUDA kernel（self 用 PyTorch gather、vLLM 用 CUDA）≈ 30% 差距；CUDA graph + scheduling 优化 ≈ 20% 差距；累计 ~3×

11. **如果只能再做一项优化，做什么？预期收益多少？**
    - 想点：接 FlashAttention（pip install flash-attn 替换 SDPA）；预期 attention compute 2-3× → 整体吞吐 +50-80%；工程量 2 天；ROI 最高

12. **同硬件 + 同模型 + 同 workload 是公平对比的关键——任意一项不同结论不可信。常见踩坑？**
    - 想点：(a) 三个 engine 不同 GPU；(b) workload 用了不同数据集；(c) sampling 配置不一致（self greedy + vLLM top-p 不可比）；(d) bench 时其他 GPU 占用；(e) 一个 engine 用 fp8 一个 bf16

---

## 串联（B 课程整体）

13. **B-M0 → B-M4 串起来，你能讲清楚的 LLM 推理优化故事是什么？**
    - 想点：(1) 从 naive batched 起步、TTFT/TPOT 分开测；(2) continuous batching 解 padding waste + head-of-line blocking、吞吐 2×；(3) PagedAttention 解 fragmentation、并发数 1.5×；(4) Tensor parallel 跨多卡 / 多机；(5) 与 vLLM 对比、归因到 FlashAttention / CUDA graph / 量化、知道下一步先做什么

14. **如果面试官问"你的 engine 离生产可用差多远"，你怎么答？**
    - 想点：诚实——demo grade、不是生产；缺 (1) FlashAttention kernel；(2) CUDA graph；(3) 量化；(4) PagedAttention CUDA kernel；(5) 容错 / 监控 / autoscaling；但**核心机制吃透**了——continuous batching 调度、PagedAttention 数据结构、TP 通信都能写、能讲

---

## 自查标准

- 14 题里 ≥ 11 题 30 秒内答得出 → B-M4 过关 → 课程 B 完成
- 题 1-3（FlashAttention）和 题 13-14（串联讲故事）必答好
