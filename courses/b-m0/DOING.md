# B-M0 · Doing 任务详细规格

> Topic：LLM 推理基础——prefill (compute-bound) vs decode (memory-bound)、serving 指标 (TTFT/TPOT/throughput)、sampling 算法。这是 B 课程后续所有优化的对照基线。
>
> 三个任务：`naive_infer.py` = baseline + 指标测量；sampling 实现 = 验证概率分布；`baseline_bench.md` = 固定 workload 数字以备后续对比。

### 模型权重来源（B 课程贯穿）
- **(a) HF GPT-2 small（124M）** — 推荐 B-M0 起步：标准 MHA + 绝对位置编码、最简单
- **(b) 课程 A 自训权重** — 可选：端到端故事更好讲；非必须
- **(c) Llama-2-7B / Qwen / Mistral-7B** — B-M3/B-M4 看 production-size 时切到这一档（先在 B-M0/B-M1/B-M2 用 GPT-2 把逻辑写对，再切大模型）

---

## 任务 1 · `naive_infer.py` — Baseline 推理 + TTFT/TPOT 测量

### 为什么做这个
LLM 推理为什么慢、贵成这样？关键在于它分两个阶段：prefill（一次性吞掉整段 prompt）和 decode（一个 token 一个 token 往外吐）。这两个阶段的瓶颈完全不同——prefill 是算力打满（compute-bound），decode 是被显存带宽卡住（memory-bound）。你要先亲手把它们分开测出来（TTFT 是 prefill 的时间，TPOT 是 decode 每步的时间），才能真正看懂后面所有优化（continuous batching、PagedAttention、tensor parallel）到底在治哪个病。这是整条推理优化线的体温计。

### 目标
写一个最朴素的 batched inference：用 HuggingFace `AutoModelForCausalLM` 加载 GPT-2 small，跑 batch=1/8/32 generation，分开测 TTFT 和 TPOT，输出吞吐（tokens/sec）。

### Sub-tasks
1. **加载模型**：
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   tok = AutoTokenizer.from_pretrained("gpt2")
   model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.bfloat16).to("cuda").eval()
   ```
   `torch_dtype=torch.bfloat16` + `model.eval()` + 推理时 `with torch.no_grad():` 三件套必备

2. **构造 batch**：固定 prompt 集合（如 8 段 ~50 token 的英文）；用 `tok(prompts, padding=True, return_tensors="pt").to("cuda")` 拿 input_ids + attention_mask；padding 用 left-pad（generation 默认要求左 padding）

3. **手写 generation 循环（不用 `model.generate()`）**：
   ```python
   # prefill: 一次性算所有 prompt token
   with torch.no_grad():
       out = model(input_ids, attention_mask=mask, use_cache=True)
   past = out.past_key_values  # KV cache
   next_logits = out.logits[:, -1, :]
   first_token = greedy_or_sample(next_logits)
   
   # decode loop: 每步一个 token
   for step in range(max_new_tokens - 1):
       out = model(first_token.unsqueeze(-1), past_key_values=past, use_cache=True)
       past = out.past_key_values
       next_logits = out.logits[:, -1, :]
       next_token = greedy_or_sample(next_logits)
       first_token = next_token
   ```
   注意：手写不是为了"造轮子"——是为了 **TTFT (prefill 时间) 和 TPOT (decode 每步时间) 必须分开测**

4. **指标测量**：
   - `torch.cuda.Event` 计 GPU 时间，不要 `time.time()`
   - **TTFT** = prefill 时间（输入到第一个生成 token 产出）
   - **TPOT** = decode 阶段平均每步时间（不含 prefill）
   - **Throughput** = `total_generated_tokens / total_time`（包含 prefill）
   - 跑 batch_size = 1, 8, 32（看 GPU 显存够不够；OOM 就降 max_new_tokens）

5. **打表**：固定 prompt_len=50, max_new_tokens=128，跑 5 次取 median；输出表格：

   | batch | TTFT (ms) | TPOT (ms/token) | Throughput (tok/s) |
   |---|---|---|---|
   | 1 | ... | ... | ... |
   | 8 | ... | ... | ... |
   | 32 | ... | ... | ... |

### 成功标准
- 三个 batch 数字都给出
- TTFT 随 batch 增长几乎不变（最多 1.5×）→ 说明 prefill 是 compute-bound、把 batch 一起喂吃满 GPU
- TPOT 随 batch 增长很慢（如 1×→32×, TPOT 涨 < 4×）→ 说明 decode 是 memory-bound、batch 摊平 KV cache 读
- Throughput 几乎线性扩展（batch=32 接近 batch=1 的 16-30×）→ 说明 batch 是 LLM serving 的主要优化抓手

### 失败排查
- **TTFT 随 batch 几乎线性增长**：说明 GPU 没吃饱（model 太小 + batch 又小）；切到大模型 / 大 batch 再测
- **TPOT 比 TTFT 还大**：你算错了——TPOT 是单 token 时间，TTFT 是整个 prompt 的时间；TPOT 应该是 ms/token 数量级（GPT-2 + A10G ~5-15 ms）
- **batch=32 OOM**：KV cache 占用 = `n_layer × 2 × batch × n_head × seq × head_dim × bytes`；GPT-2 12 layer × 12 head × 64 dim × bf16 × batch 32 × seq 200 ≈ 75 MB——不应该 OOM；OOM 大概率是没 free 中间 activation 或者 dtype 还是 fp32

### 辅助阅读（非 canonical）
- HF `model.forward(use_cache=True)` 文档：https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel
- "Mastering LLM Techniques: Inference Optimization" (NVIDIA blog)：https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

### Deliverable
- `course-b/m0-baseline/naive_infer.py`
- `course-b/m0-baseline/baseline_results.md`：硬件 + 模型 + 表格 + 一段观察

---

## 任务 2 · Sampling 实现 + 单测验证

### 为什么做这个
模型每步吐出来的不是一个 token，而是一整个词表上的概率分布——到底挑哪个，由 sampling 决定。greedy、temperature、top-k、top-p 这几种策略直接决定输出是死板还是发散，也是后面 engine 必不可少的零件。自己手写一遍，你才会知道这些参数（比如 temperature 调高调低）背后到底在动概率分布的什么地方，而不是把它们当成玄学旋钮。用单测验证分布行为，是为了确认你写的逻辑真的符合数学预期，而不是"看起来差不多"。

### 目标
手写 greedy / temperature / top-k / top-p sampling，用单测验证概率分布行为。这是后续 engine 的必备组件。

### Sub-tasks
1. 写 `sampling.py`，每个函数签名 `(logits: Tensor [B, vocab]) -> Tensor [B]` 返回 token id：
   - `greedy(logits)`：`logits.argmax(dim=-1)`
   - `temperature_sample(logits, T)`：`F.softmax(logits / T, dim=-1)` → `torch.multinomial(probs, 1).squeeze(-1)`；T → 0 等价 greedy；T → ∞ 接近均匀分布
   - `top_k_sample(logits, k)`：保留 logits 最大的 k 个、其他设 -inf；softmax + multinomial
   - `top_p_sample(logits, p)`：sort logits descending、cumulative softmax、找到第一个累积 ≥ p 的位置 + 1，截断后续设 -inf
2. 写 `tests/test_sampling.py`：
   - **greedy 确定性**：同一 logits 跑 100 次结果完全一致
   - **temperature**：T=0.001（接近 greedy）vs T=10（接近均匀）→ 跑 1000 次 sample 算分布熵；T=0.001 熵 ≈ 0、T=10 熵接近 log(vocab_size)
   - **top-k**：k=5 → 跑 1000 次 sample，确认所有结果都在 logits top-5 里
   - **top-p**：p=0.9 → 跑 1000 次 sample，确认所有结果对应 logit 在累积概率前 90% 内

### 成功标准
- 4 个 sampling 函数 + 4 组单测都过
- 单测涉及随机性的用 `torch.manual_seed(0)` 锁住

### 失败排查
- **temperature → 0 报错**：`logits / 0` 是 NaN；clamp 到 `T = max(T, 1e-5)` 或 `T == 0` 直接 fallback 到 greedy
- **top-p 边界**：当所有 token 第一个就累积超过 p 时要保留至少 1 个 token（写代码时容易误删全部）
- **top-k 与 top-p 单独单测过、组合调用挂**：通常是 in-place 修改 logits 互相影响；clone 一份再改

### 辅助阅读（非 canonical）
- HF `LogitsProcessor` 源码（参考实现，**不要直接抄**）：https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
- "The Curious Case of Neural Text Degeneration"（top-p 论文）：https://arxiv.org/abs/1904.09751

### Deliverable
- `course-b/m0-baseline/sampling.py`
- `course-b/m0-baseline/tests/test_sampling.py`

---

## 任务 3 · `baseline_bench.md` — 固定 workload 对照基线

### 为什么做这个
后面每做一个优化，你都得回答一个问题：到底快了多少？如果每次用的 prompt 和参数都不一样，那数字根本没法比，等于白测。所以现在要把一份"固定的 prompt 集 + 固定输出长度"冻起来，跑出一组 baseline 数字当锚点，之后所有模块（continuous batching、PagedAttention 等）都对照这同一份 workload 跑。顺便，这一步用最朴素的 static batching 把短 prompt 全 pad 到最长，你会亲眼看到 padding 浪费了多少算力——这正是下一个模块 continuous batching 要解决的问题。

### 目标
冻一份"固定 prompt 集 + 固定输出长度"的 workload，跑出 baseline 数字。这是后续 B-M1 / B-M2 / B-M3 / B-M4 所有优化对比的锚点——所有优化都要对照这个跑数字。

### Sub-tasks
1. **冻结 workload**：
   - 一组 prompts（建议 100 条，长度 50-500 token 不等模拟真实分布）；保存为 `workload.jsonl`
   - 固定 `max_new_tokens = 128`
   - 固定 sampling：greedy（消除 sampling 噪声）
2. **跑 baseline**（用任务 1 的 `naive_infer.py`）：
   - batch=1（逐条跑）
   - batch=N（一次喂 N 条，但 padding 到最长——这就是 static batching 的 padding waste 演示）
3. **记录到 `baseline_bench.md`**：
   - 硬件（GPU 型号 + 显存）
   - 模型（gpt2 / 自训 / 其他）
   - workload 描述（条数 + 长度分布）
   - 表：(batching mode, TTFT_p50, TTFT_p99, TPOT_p50, throughput_total)
   - **画出 padding waste**：batch=N 时实际 compute / 有效 compute 的比例（短 prompt 被 pad 到最长那条）

### 成功标准
- 表格四列数字给齐
- batch=N 的吞吐显著高于 batch=1（说明 batch 有用），但比"理论上线性 N 倍"差不少（说明 padding waste + decode 还是 memory-bound）
- README 一段话：定义"什么是 padding waste"——这是 B-M1 continuous batching 要解决的问题

### Deliverable
- `course-b/m0-baseline/workload.jsonl`
- `course-b/m0-baseline/baseline_bench.md`

---

## 三个任务做完之后

- 跑 QUIZ.md
- B-M0 过关 → 开 B-M1（KV cache + continuous batching）
