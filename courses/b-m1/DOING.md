# B-M1 · Doing 任务详细规格

> Topic：手写 KV cache 数据结构 + continuous batching 调度器。理解 static batching 的 padding waste / head-of-line blocking 和 continuous batching 的解法。
>
> 三个任务：`engine_v1/` 完整 engine（kv_cache + scheduler + engine）/ 1k 请求 stress test / `cb_vs_static.md` 对照。

### 模型选择
继续用 GPT-2 small（B-M0 同款）——B-M1 重点是调度逻辑，不是大模型；逻辑写对再切大模型。

---

## 任务 1 · `engine_v1/` — KV cache + Scheduler + Engine 主循环

### 为什么做这个
B-M0 已经看到 static batching 的毛病：一批请求必须等齐才能开跑、短的被长的拖着、显存全被 padding 占着。真正的 serving 系统（vLLM 那一类）是怎么解的？答案是把调度从"按 batch"细化到"按每一步 iteration"——谁算完了立刻让出位置，新来的请求随时插进来填空。要理解这套机制，光读源码没用，得自己写一个最小版本：一个预分配的 KV cache（绝不每步 cat，那是性能毒药）、一个按 iteration 调度的 scheduler、一个主循环。把这三块拼起来跑通，你就真正摸到了现代推理引擎的骨架。

### 目标
不依赖 HF `model.generate()`，自己写一个 minimal serving engine：暴露 `add_request(prompt, max_tokens) -> request_id`，主循环每 iteration 调度新 request 的 prefill / 已有 request 的 decode、释放完成 request 的 KV slot。

### Sub-tasks

#### 1.1 `kv_cache.py` — 预分配 KV cache + slot 管理
- 数据结构：一个大 tensor `kv_cache[n_layer, 2, max_batch, n_head, max_seq_len, head_dim]`，2 是 K/V
- 一次性预分配（**绝不 `torch.cat`** 每步增长——会触发 allocator 重新申请 + memcpy）
- 维护 `slot_state: List[None | RequestId]` 长度 `max_batch`，None 代表空 slot
- API：
  - `allocate(request_id) -> slot_id | None`：找一个空 slot 占用、返回 id；满了返 None
  - `free(slot_id)`：释放 slot，对应位置 KV 不需清零（下次写覆盖）
  - `append_kv(slot_id, layer, k_new, v_new, position)`：写入 `kv_cache[layer, 0, slot_id, :, position, :] = k_new`（**写入而不是 cat**）
  - `get_kv(slot_id, layer, end_position)`：返回 `kv_cache[layer, :, slot_id, :, :end_position, :]` 给 attention 用

#### 1.2 `scheduler.py` — Iteration-level scheduler
- 每个 request 状态：`WAITING / PREFILL / DECODE / FINISHED`
- 维护三个队列：`waiting_queue`（新来的）、`running_set`（已分配 slot）、`finished_list`
- `schedule()` 每 iteration 调用，返回当 iteration 要 forward 的 (prefill_requests, decode_requests)：
  ```
  - 尝试从 waiting_queue 拉一个 request：
    - kv_cache.allocate() 拿 slot；满了就停
    - 算 KV 预算够不够（prompt_len + max_tokens 不能超 max_seq_len）
    - 加入 prefill_requests
  - 已经在 running_set 的（PREFILL 完成进了 DECODE 的）加入 decode_requests
  - 已 FINISHED（达到 max_tokens 或 EOS）→ kv_cache.free()，移到 finished_list
  ```
- **prefill 和 decode 不能简单 concat 进同一 batch**：attention mask 不同（prefill lower triangular、decode 单 token）；推荐 (a) **分开 forward 调用**（同 step 内先跑 prefill 一批、再跑 decode 一批，两次 kernel 调用），不推荐 (b) 变长 attention（FlashAttention 风格，B-M1 不要求）

#### 1.3 `engine.py` — Engine 主循环
- API：
  - `add_request(prompt: str, max_tokens: int) -> request_id`（返回 id 给 client，client 用 id 拉结果）
  - `step()`：跑一个 iteration（schedule + forward + sample + 返回新生成的 token）
  - `get_result(request_id) -> RequestResult | None`（poll 完成结果）
- 主循环（在 thread 或 async 都行；先单线程）：
  ```
  while True:
      prefill_reqs, decode_reqs = scheduler.schedule()
      if not prefill_reqs and not decode_reqs:
          if no pending request: break
          else: continue
      
      if prefill_reqs:
          run_prefill_batch(prefill_reqs)  # 一次 forward 算所有 prompt token、写 KV
      if decode_reqs:
          run_decode_batch(decode_reqs)  # 一次 forward 算每个 request 的下一个 token
      
      sample tokens, update request states, free finished slots
  ```

#### 1.4 整合 GPT-2
- 替换 HF `model.generate()` 用上面的 engine
- 因为 HF GPT-2 的 KV cache 接口和你的不一样，**只用 model 的 forward**：
  - prefill：`model(input_ids, attention_mask=mask, use_cache=False)`，自己从 hidden states 算 K/V 写进你的 cache（这一步要 hack 一下、或者每层 forward 后用 hook 抓 K/V）
  - 简化做法：直接让 `forward` 接收 `past_key_values` 参数（HF 接口本身支持）；你的 `kv_cache.get_kv()` 输出的就是 `past_key_values` 格式 → 不需要 hack model

### 成功标准
- `add_request` 1k 个请求（prompt 长度均匀分布 50-500、max_tokens 均匀分布 32-256）跑完不挂
- 输出 token 与 baseline `naive_infer.py` 在 greedy sampling 下逐字一致（验证逻辑正确）
- KV slot 在 request 完成后被复用（看 logging：slot 0 用过、free 了、又被新 request 用了）

### 失败排查
- **输出和 baseline 不一致**：position 算错了（KV cache append 时位置应该是已有 token 数 + 当步新 token 偏移）；或者 attention mask 不对（prefill 是 causal、decode 是只看 cache）
- **`torch.cat` 触发 OOM**：你确实在 append KV 时用 cat 了——改成预分配 + 写入
- **同一 slot 被多个 request 抢**：scheduler 线程不安全；或者 free 和 allocate 的状态机有竞争——单线程版应该没问题，加 lock 或检查代码

### 辅助阅读（非 canonical）
- vLLM scheduler 源码（带着读、画状态机）：https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py
- HF transformers `past_key_values` 格式说明：https://huggingface.co/docs/transformers/main_classes/output

### Deliverable
- `course-b/m1-engine/engine_v1/{kv_cache.py, scheduler.py, engine.py}`
- 一个 `tests/test_correctness.py`：100 条 prompt 的 engine 输出 vs `naive_infer.py` 输出逐字对比

---

## 任务 2 · 1k 请求 stress test + 吞吐对比

### 为什么做这个
写出来的 engine 到底比 B-M0 的 static batching 快多少？不能凭感觉，得拿 1000 个长短不一的真实请求灌进去，跟 baseline 同台对比吞吐和延迟。这一步逼你直面 continuous batching 真正的收益来源：它不靠单个请求跑得快，而是靠"GPU 一刻不闲"——完成的请求立刻释放、新请求立刻补位。如果数字没到预期的 2 倍提升，你还得分析为什么（模型太小？调度开销太大？prompt 长度方差不够？），这种归因能力本身就是理解 serving 系统的核心。

### 目标
跑 1000 个随机 request（混合 prompt/output 长度），对比 continuous batching engine vs B-M0 的 static batching baseline。

### Sub-tasks
1. **生成 workload**：复用 B-M0 的 `workload.jsonl`（如果有）或新生成 1000 条：prompt_len uniform[50, 500]、max_tokens uniform[32, 256]
2. **跑 baseline (B-M0 static batching)**：把 1000 条按 batch=32 切成 32 批，每批 padding 到最长，跑完所有批；记 total_time、p50/p99 TTFT、throughput
3. **跑 engine_v1 (continuous batching)**：1000 条都 `add_request` 进去，让 scheduler 自己 batching；记 total_time、p50/p99 TTFT、throughput
4. **对比表**：

   | metric | static (baseline) | continuous (engine_v1) | improvement |
   |---|---|---|---|
   | total_time | ... | ... | ... |
   | throughput (req/s) | ... | ... | ... |
   | TTFT p50 | ... | ... | ... |
   | TTFT p99 | ... | ... | ... |

### 成功标准
- continuous batching 吞吐 ≥ **2×** static baseline
- TTFT p99 也降（不会一个长 prompt 拖死后面所有 request）
- 如果不到 2×，写一段原因分析（常见：模型太小、scheduling overhead 占比大；或者 prompt 长度方差不够大、padding waste 不明显）

### 失败排查
- **continuous 比 static 还慢**：scheduler overhead 太大（每 iteration 重排所有队列）；或者每 iteration 只调度 1 个 request（没真正 batch 起来）
- **TTFT p99 没降**：长 prompt 还是把整个 batch 卡住——这是预期，B-M1 的简单实现里"prefill 一个长 prompt 占满整个 step"会挡住 decode；需要 chunked prefill（B-M2 / vLLM 的进阶话题）才能彻底解
- **OOM**：max_batch 太大；KV cache 预算超过 GPU 显存；调小 max_batch

### Deliverable
- `course-b/m1-engine/stress_test.py`
- `course-b/m1-engine/cb_vs_static.md`（对比表 + 解释）

---

## 任务 3 · `cb_vs_static.md` 详细对照

### 为什么做这个
有了数字还不够，你得能讲清楚"为什么 continuous batching 赢"——这是把一堆实验变成一个能讲给别人听的故事。这一步用 timeline 图把两种调度的差别画出来：static batching 里 GPU 经常空转等最慢的请求，continuous batching 里 GPU 一直有活干。同时把两个核心病灶讲明白：padding waste（短请求被填充到最长）和 head-of-line blocking（一个长请求把后面所有短请求堵死）。能把这两件事讲透，说明你不只是会写代码，而是真懂了这套优化在治什么病。

### 目标
扩展任务 2 的对照成完整的"为什么 continuous batching 赢"分析，作为面试可讲的故事。

### Sub-tasks
1. 任务 2 的对比表
2. 用 timeline 图（手画也行，markdown ASCII art 也行）展示：
   - **Static batching**：32 个 request 一起开始，最长那个跑完前其他都被卡住；中间 GPU 经常 idle
   - **Continuous batching**：每 iteration 释放完成的 slot、立刻塞新 request 进来；GPU 一直有事干
3. 一段分析：
   - **Padding waste**：static 在每 batch 里把短 prompt pad 到最长 → compute 和 KV 浪费
   - **Head-of-line blocking**：static batch 里如果有一个 max_tokens=512 的 request、其他 max_tokens=32 的请求被它锁住等 480 step
   - **Continuous batching 解了什么**：(a) 不需要等齐 batch、(b) 完成的立刻释放、(c) 新来的立刻接入；代价是 scheduling overhead

### Deliverable
- `course-b/m1-engine/cb_vs_static.md`（最终版）

---

## 三个任务做完之后

- 跑 QUIZ.md
- B-M1 过关 → 开 B-M2（PagedAttention 显存优化）
