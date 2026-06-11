# B-M0 · 知识自查口试

> 验证 prefill vs decode 资源差异 / TTFT/TPOT 定义 / sampling 数学是否吃透。30 秒内答得出 = 过。

---

## Prefill vs Decode（对应任务 1）

1. **Prefill 阶段 1×1024 token 的 attention 计算量是 decode 每步的多少倍？为什么？**
   - 想点：prefill 一次算 1024 个位置的 attention（QKV 都是 [1024, head_dim]），attention 矩阵是 [1024, 1024]、O(N²) ≈ 1M 元素；decode 每步 Q 是 [1, head_dim]、K/V 是 [N, head_dim]、attention 矩阵 [1, N]、O(N) ≈ 1024 元素；prefill 是 decode 单步的 ~1024 倍 attention compute

2. **为什么 prefill 是 compute-bound、decode 是 memory-bound？**
   - 想点：prefill 大矩阵乘吃 FLOPs，GPU 算力满载；decode 每步只算 1 个 Q × N 个 K/V，FLOPs 少但要把整个 KV cache 从 HBM 读出来（每层 2 × N × n_head × head_dim × bytes）→ 瓶颈是 HBM 带宽不是算力

3. **batch=32 比 batch=1 在 prefill 阶段时间几乎不变，为什么？decode 阶段呢？**
   - 想点：prefill compute-bound、本来 GPU 没吃满 → batch 起来 = 几乎免费 throughput；decode memory-bound、KV cache 在 batch 维 stack 起来读、HBM 带宽是瓶颈 → batch 涨 32× 时间最多涨 ~5-10×（KV 读其实是每个 sample 独立的、并不能完全摊平）

4. **如果模型从 7B 升到 70B，prefill 和 decode 各变多慢？**
   - 想点：prefill 吃算力 → ~10× 慢；decode 吃 HBM 带宽 → KV cache 也变 10× → 也 ~10× 慢（参数量和 KV cache 量都 ~ 模型大小）

---

## Serving 指标（对应任务 1 + 任务 3）

5. **TTFT 和 TPOT 各是什么？为什么必须分开测？**
   - 想点：TTFT = 收到 request → 第一个 token 产出（≈ prefill 时间 + 第一次 decode）；TPOT = decode 阶段每个 token 平均时间；分开测 = 因为 prefill 和 decode 是两种 workload，瓶颈不同

6. **总 latency / total_tokens 是错的指标——为什么？**
   - 想点：把 prefill 时间也平摊到每个生成 token，掩盖了"用户等多久看到第一个字"和"后续输出多流畅"是不同体验；产品视角：TTFT 影响体验起步、TPOT 影响后续流畅度

7. **p99 latency 比 p50 重要吗？serving 系统优化哪个？**
   - 想点：都重要、看场景；产品体验对 p99 敏感（少数请求慢 5 倍用户会骂）；优化 p99 要看尾巴是被什么拖的（长 prompt prefill / 长输出 decode / 排队等待）；vLLM 的 continuous batching 主要降 p99（避免 head-of-line blocking）

---

## Sampling 数学（对应任务 2）

8. **temperature 是怎么调节随机性的？T=0 和 T=∞ 各是什么效果？**
   - 想点：`softmax(logits / T)`；T=0 → 概率集中在 argmax → 等价 greedy；T=∞ → 所有概率趋同 → 均匀采样；T=1 是原始分布

9. **top-k=50 vs top-p=0.95 哪种更鲁棒？**
   - 想点：top-k 不管概率分布、永远保留 k 个候选（可能保留低概率的）；top-p 自适应——分布尖锐时只保留 1-2 个、平坦时保留几十个；top-p 一般更鲁棒；常配合用：`top_k=50` 先粗过滤 + `top_p=0.95` 再精过滤

10. **greedy 在生产环境为什么不常用？**
    - 想点：(1) 没多样性（同 prompt 永远同输出）；(2) 容易 mode collapse（生成同字符的 loop）；(3) 评估有用（可重复）但部署用 sampling

---

## KV Cache 直觉（B-M1 伏笔）

11. **`naive_infer.py` 里 `past_key_values` 是什么？没有它会怎样？**
    - 想点：HF transformers 的 KV cache 接口；每次 forward 把上一步的 K/V 传进去、当步只算新 token 的 K/V append 到 cache；不传 → 每步都把前面所有 token 重算一遍 K/V → O(N²) 慢爆

12. **KV cache 占多少显存？写一个公式。**
    - 想点：`n_layer × 2 × batch × n_head × seq_len × head_dim × bytes`；GPT-2 small bf16 batch=32 seq=1024：12 × 2 × 32 × 12 × 1024 × 64 × 2 ≈ 1.2 GB——已经接近模型本身的 ~1.5×

13. **batch=N 的 padding waste 是什么？continuous batching 怎么解？**
    - 想点：static batching 把短 prompt pad 到最长那条 → padding 位置也在算 compute、也占 KV cache；continuous batching 让每个 request 独立调度、不需要等齐其他 request；这是 B-M1 的核心

---

## 自查标准

- 13 题里 ≥ 11 题 30 秒内答得出 → B-M0 过关
- 题 1-4（prefill vs decode）必答好——这是 B 课程的根基
