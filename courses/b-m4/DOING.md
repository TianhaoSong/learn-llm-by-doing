# B-M4 · Doing 任务详细规格

> Topic：自己的 engine vs vLLM vs SGLang 三方对比；把差距归因到具体机制；给出优先级清单。线 2 终点。
>
> 三个任务：API 兼容 + 起三个服务 / 三方 benchmark / 归因报告 + 优先级。

### 模型选择
Llama-2-7B（B-M3 同款）；如能跑 13B 就 13B，差距更明显。

---

## 任务 1 · self-engine 暴露 OpenAI API + 起 vLLM / SGLang

### 为什么做这个
要公平比较你的 engine 和 vLLM / SGLang，三家必须用同一把尺子量。业界的事实标准是 OpenAI 的 API 格式——只要你的 engine 也暴露 `/v1/completions` 这样的接口，就能直接用 vLLM 自带的压测脚本一视同仁地打三家，数字才可比。这一步还顺便逼你把前面几个模块攒下来的 engine 包装成一个真正能对外提供服务的 HTTP server，而不是只能在脚本里调的玩具。把三个服务都起起来、各自能正常返回结果，是后面做三方对比的前提。

### 目标
让 vLLM 自带的 `benchmark_serving.py` 能直接打你的 engine——也就是你的 engine 必须暴露 OpenAI 兼容 API。这样三方数据可比。

### Sub-tasks

#### 1.1 self-engine 暴露 OpenAI 兼容 API
- 用 FastAPI 或 aiohttp 起 HTTP server
- 实现 `/v1/completions` endpoint（OpenAI legacy）或 `/v1/chat/completions`（新版）
  - 入参：`{"model": "...", "prompt": "...", "max_tokens": ..., "stream": false}`
  - 出参：`{"choices": [{"text": "..."}], "usage": {...}}`
- 内部调你的 `engine_v3.engine.add_request()` + poll `get_result()`
- streaming 选做（仅在你时间够时；不影响 benchmark）

#### 1.2 起 vLLM 服务
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --tensor-parallel-size 8 \
  --port 8001
```

#### 1.3 起 SGLang 服务
```bash
pip install sglang[all]
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-hf \
  --tp 8 \
  --port 8002
```

#### 1.4 三个服务都起来
- self-engine: `localhost:8000`
- vLLM: `localhost:8001`
- SGLang: `localhost:8002`
- 各 sanity check：`curl localhost:800X/v1/completions -d '{"model": "...", "prompt": "Hello", "max_tokens": 10}'`，三个都能返回合理输出

### 成功标准
- 三个服务都正常返回 generation
- self-engine 输出和 vLLM 输出在 greedy 下不要求逐字一致（不同 attention 实现数值有微小差异），但应该高度类似（前 10 token 一致就行）

### 失败排查
- **vLLM 启动 OOM**：`--max-model-len 2048` 限制 max sequence length 节省 KV cache 预算
- **SGLang 装不上**：装 `sglang[all]` 时 CUDA 版本必须匹配；用 DLAMI 的话默认 OK
- **self-engine 收到请求但卡住**：你的 engine 主循环在新 thread 里跑、API handler 等不到结果——加 `asyncio` 或者用 `time.sleep(0.01)` polling

### 辅助阅读（非 canonical）
- OpenAI API spec：https://platform.openai.com/docs/api-reference/completions
- vLLM serving docs：https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- SGLang server docs：https://github.com/sgl-project/sglang#backend-sglang-runtime-srt

### Deliverable
- `course-b/m4-compare/self_engine_server.py`
- `course-b/m4-compare/start_servers.sh`：一键起三个服务

---

## 任务 2 · 三方 benchmark

### 为什么做这个
你自己从零搭的 engine，跟工业级的 vLLM / SGLang 比到底差多远？这一步用真实的 ShareGPT 变长 workload，在同一硬件上把三家挨个压测，记下 TTFT、TPOT、吞吐和 GPU 利用率。重点不是"我居然比 vLLM 慢"——慢 3 到 5 倍完全正常、是预料之中的；重点是拿到一组诚实的数字，知道差距具体有多大、体现在哪个指标上。这组数字是下一步归因分析的原料，没有它，"我该先优化什么"就只能瞎猜。

### 目标
用 vLLM 自带的 `benchmark_serving.py` 跑同一 workload 打三个服务，记录 (TTFT, TPOT, throughput, GPU util)。

### Sub-tasks
1. **数据**：ShareGPT 子集 https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered （或自己合成 1000 条变长 prompt）
2. **bench 脚本**：用 vLLM 的 `benchmarks/benchmark_serving.py`：
   ```bash
   python benchmarks/benchmark_serving.py \
     --backend openai \
     --base-url http://localhost:800X \
     --model meta-llama/Llama-2-7b-hf \
     --dataset-name sharegpt \
     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
     --num-prompts 1000 \
     --request-rate 16
   ```
3. **三个服务各跑一次**（同一硬件、同一时间窗口、错开跑——避免互相影响 GPU），记录：
   - TTFT_p50, TTFT_p99
   - TPOT_p50, TPOT_p99
   - Throughput (tok/s 总)
   - GPU util（用 `nvidia-smi --query-gpu=utilization.gpu --format=csv` 平均值）
4. **打表**：

   | engine | TTFT p50 | TTFT p99 | TPOT p50 | Throughput | GPU util |
   |---|---|---|---|---|---|
   | self-engine | ... | ... | ... | ... | ... |
   | vLLM | ... | ... | ... | ... | ... |
   | SGLang | ... | ... | ... | ... | ... |

### 成功标准
- 三组数字都给全
- self-engine vs vLLM throughput 比例（self 应该是 vLLM 的 1/3 - 1/5；这是合理范围、不要纠结）
- GPU util self-engine 应该 < vLLM（说明有空间优化）

### 失败排查
- **self-engine 中途卡死**：常是 OOM、KV cache 用尽——`max_batch` 调小
- **vLLM 比 SGLang 慢**：SGLang 默认开 RadixAttention 对长 system prompt 有 prefix caching 收益；workload 没共享 prefix 时差不多；这是预期
- **self-engine 比 vLLM 还快**：检查是不是 self-engine 漏了某个 step（如 sampling 没跑、token 不对）；正常情况下 PyTorch 实现一定比 vLLM 慢

### Deliverable
- `course-b/m4-compare/benchmark_results.md`：表格 + 硬件 / 模型 / workload 信息

---

## 任务 3 · 归因报告 + 优先级清单

### 为什么做这个
知道"慢了多少"只是开始，真正值钱的是能回答"为什么慢、先补哪块收益最大"。这一步把任务 2 的每一项差距拆到具体机制上——是缺了 FlashAttention？没用 CUDA graph？paged attention 还在 gather？少了 prefix caching 或 chunked prefill？每项都要有差距百分比、归因机制和证据。然后据此排出一份"如果继续投入先做什么"的优先级清单，每项标上预期收益和工程量。这是整条推理线的收口：它把你前面所有动手积累，变成一个能讲给资深工程师听、有数据有判断的完整故事。

### 目标
把任务 2 的差距归因到具体机制，给出"如果继续投入会先做什么"的优先级。这是 B 课程整体的最高密度产出。

### Sub-tasks

#### 3.1 `self_vs_vllm.md` — 归因清单
对每项差距，列出：(指标, 差距 %, 归因机制, 证据)
1. **FlashAttention**：vLLM 用 FA2、self-engine 用 PyTorch SDPA
   - 影响：attention compute 慢 2-3×（特别是长 seq）
   - 证据：profiler 看 attention kernel 时间
2. **CUDA graph**：vLLM `enforce_eager=False` 默认开 CUDA graph、减少 kernel launch overhead
   - 影响：decode 阶段 step time 慢 10-20%
   - 证据：vLLM 启动时 log 会打 "capturing CUDA graphs"
3. **PagedAttention CUDA kernel**：vLLM 在 block 上直接算 attention、不 gather；self-engine PyTorch 实现要 gather
   - 影响：每 step 慢 30-50%
   - 证据：B-M2 已经在 mem_efficiency.md 提过
4. **Prefix caching**：vLLM v0.4+ 默认开（automatic prefix caching）；ShareGPT 共享 system prompt 时显著
   - 影响：TTFT_p50 vLLM 显著低
   - 证据：vLLM 启动 log
5. **Continuous batching scheduler 优化**：vLLM 的 chunked prefill、recompute、preemption 等
   - 影响：长 prompt 不卡住其他 request、TTFT_p99 vLLM 优势明显
6. **量化**（如果 vLLM 用 fp8/awq）：本对比中 self-engine 用 bf16；如 vLLM 也 bf16 这项不算

#### 3.2 `next_steps_priorities.md` — 3 项优化优先级
根据归因排序，列 top 3：
1. **P0**: 接入 FlashAttention（用 `flash-attn` pip package，替换 SDPA）
   - 预期收益：attention compute 快 2-3×，整体吞吐 +50-80%
   - 工程量：~2 天（接 attention API、benchmark 验证）
2. **P1**: CUDA graph capture（PyTorch `torch.cuda.CUDAGraph`）
   - 预期收益：decode 阶段每 step 减少 1-3 ms kernel launch overhead，TPOT -10-20%
   - 工程量：~3 天（要求 input shape 固定、PagedAttention 配合）
3. **P2**: chunked prefill（长 prompt 切片、和 decode 错峰）
   - 预期收益：TTFT_p99 显著降低（长 prompt 不再卡住）
   - 工程量：~5 天（scheduler 大改）

### 成功标准
- `self_vs_vllm.md` 至少列 5 项归因 + 每项有"差距 %"和"机制"和"证据"
- `next_steps_priorities.md` 列 3 项优先级 + 每项有"预期收益"和"工程量"

### 失败排查
- **没法定量归因**：profiler trace 抓 self-engine vs vLLM 同 workload 一段；看 attention / kernel launch / scheduling 各占多少
- **不知道 vLLM 内部用了什么**：vLLM 启动 log 会打主要 feature flag；或者读 vLLM 的 release notes

### Deliverable
- `course-b/m4-compare/self_vs_vllm.md`
- `course-b/m4-compare/next_steps_priorities.md`

---

## 三个任务做完之后

- 跑 QUIZ.md
- B-M4 过关 → 课程 B 完成 → 串联到课程 C（agent）
