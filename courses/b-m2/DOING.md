# B-M2 · Doing 任务详细规格

> Topic：用 block-based 分页 KV cache 解决 B-M1 连续 cache 的 fragmentation + over-provisioning。本模块**只看显存利用率**——throughput 留给 B-M4 与 vLLM 对比时归因。
>
> 三个任务：`engine_v2/` 改造 + paged attention 实现 / 显存对比 / 选做 Triton kernel。

---

## 任务 1 · `engine_v2/` — Block manager + Paged attention

### 为什么做这个
B-M1 的 KV cache 是一整块连续显存，每个请求按"最大可能长度"提前占满——可它实际可能只生成几十个 token，剩下的全空着浪费，这就是 over-provisioning 和碎片化。PagedAttention 借用了操作系统虚拟内存的思路：把 KV cache 切成固定大小的小块（block），请求写满一块才申请下一块，用一张映射表把"逻辑上连续的序列"对应到"物理上零散的块"。这是 vLLM 显存效率高的根本原因，也是这整个领域最经典的一个 idea。自己实现一遍 block manager 和在分页结构上算 attention，你才会真懂虚拟内存这套思想是怎么搬到 GPU 上来的。

### 目标
把 B-M1 engine 的 KV cache 从"连续 [max_batch, max_seq_len]"改成"block-based [total_blocks, block_size]"，维护 logical-to-physical 映射。

### Sub-tasks

#### 1.1 `block_manager.py`
- 物理布局：一个大 tensor `physical_blocks[total_blocks, n_layer, 2, n_head, block_size, head_dim]`
  - `total_blocks` 通常 ≈ GPU 显存预算 / 每 block 字节数 - 模型 - activation；如 GPU 24GB，模型 + 其他 8GB → 16GB / (n_layer × 2 × n_head × block_size × head_dim × bytes) = 几千 blocks
  - `block_size = 16` token（vLLM 默认值，经验最优）
- 数据结构：
  - `free_blocks: set[int]`（可用 block id 集合）
  - `block_table: dict[seq_id, list[int]]`（每个 sequence 占用的 block id 列表）
- API：
  - `allocate(seq_id, num_blocks)`：从 free_blocks 拿 num_blocks 个、加入 block_table[seq_id]；不够返 None
  - `free(seq_id)`：把 block_table[seq_id] 的 block 全还给 free_blocks
  - `append_slot(seq_id)`：当 sequence 长度达到当前最后一 block 边界、申请新 block；返回新 block id 或 None
  - `get_block_table(seq_id) -> list[int]`：给 attention 用

#### 1.2 `paged_attention.py` — PyTorch 实现
- 输入：`q [B, n_head, 1, head_dim]`、batch 中每个 seq 的 `block_table` 和 `seq_len`
- 步骤（最直接的 PyTorch 写法）：
  ```python
  # 把每 seq 的 block 拼回连续 K/V tensor
  for seq_idx in range(B):
      blocks = block_table[seq_idx]  # list[int]
      seq_len_i = seq_lens[seq_idx]
      # gather: physical_blocks[blocks] -> [num_blocks, n_layer, 2, n_head, block_size, head_dim]
      gathered = physical_blocks[blocks]  # 或 torch.index_select
      # reshape to [n_layer, 2, n_head, num_blocks * block_size, head_dim], 截到 seq_len_i
      k_seq = gathered[:, 0, :, :seq_len_i, :]  # 简化版，实际要更细
      v_seq = gathered[:, 1, :, :seq_len_i, :]
      # 算 attention: q[seq_idx] @ k_seq.T / sqrt(d) → softmax → @ v_seq
  ```
- **关键**：这个实现**单 op 比 B-M1 连续 cache 慢 30-50%**——因为 `torch.index_select` 触发 indirect memory access、每步都搬数据；这是预期，不是 bug
- vLLM 的 CUDA kernel 直接在 block 上算（pointer dereferencing 在 kernel 内）、不拼回——所以快；你**不需要**写 kernel（Triton 选做）

#### 1.3 集成到调度器
- 改造 B-M1 的 `engine_v1` → `engine_v2`：
  - `kv_cache.py` 替换成 `block_manager.py`
  - `attention` 替换成 `paged_attention.py`
  - scheduler 的 OOM 处理：当 `block_manager.allocate()` 返回 None 时，简单**拒绝该 request 加回 waiting_queue**（B-M2 不要求实现 preemption / swap to CPU）

### 成功标准
- engine_v2 跑通 B-M1 的 1k 请求 stress test（输出与 B-M1 一致 / 与 baseline `naive_infer.py` 一致）
- 调试时打印 `len(free_blocks)` 看 block 释放是否正常（finished request 后 block 数应该回升）
- Code review checkpoint：
  - block_manager 的 free_blocks 用 `set` 不是 `list`（O(1) pop/add）
  - `block_size = 16` 写死（不 hardcode 别的数字，按 vLLM 默认）
  - paged attention 不能有显式 `torch.cat` 拼 KV（如有也要在文档里说明为什么这步必要）

### 失败排查
- **engine_v2 输出和 baseline 不一致**：通常是 `block_table` 索引到 K/V 的位置算错了——某 token 在第 N 个 block 的第 N % block_size 个位置；写一个单元 test 用 1 个 prompt 比对每层 attention 输出
- **paged attention 比连续 attention 慢 100×**：每 step `physical_blocks[blocks]` 触发 GPU 端 indirect copy；如果跑得动就不管（B-M2 过关不看 throughput），如果连一个 step 都跑不动说明 gather 写错了（比如把整个 `physical_blocks` copy 而不是 index）
- **OOM 在中间某个 step**：单 step 内同时活跃的 sequence 太多 → 调度器一次拉 prefill 时算清楚 block 预算

### 辅助阅读（非 canonical）
- vLLM `block_manager.py` 源码：https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py
- vLLM `paged_attn.py` 的 host-side（不要陷入 CUDA 细节）：https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/paged_attn.py
- "Mastering LLM Techniques: Inference Optimization" PagedAttention 章节（NVIDIA）

### Deliverable
- `course-b/m2-paged/engine_v2/{block_manager.py, paged_attention.py, engine.py}`
- 一个 `tests/test_correctness_v2.py`：100 prompt 的 v2 输出 vs baseline 一致

---

## 任务 2 · 显存效率对比 — `mem_efficiency.md`

### 为什么做这个
PagedAttention 的卖点不是"跑得快"（PyTorch 版反而更慢），而是"同样的显存能塞下更多并发请求"。这个收益必须用数字说话：在固定的显存预算下，连续 cache 的 v1 因为按最大长度预留、只能塞这么多请求；而按需分块的 v2 能塞明显更多。这一步就是把这个差距量化出来，并讲清两种浪费的来源——内部碎片（每个序列内部留白）和过度预留（按 max_tokens 而不是实际长度算）。搞清楚这笔账，你就明白了为什么显存利用率是 LLM serving 里最值钱的指标之一。

### 目标
量化"PagedAttention 的显存收益"——在固定 GPU 显存预算下，能并发塞下多少请求。

### Sub-tasks
1. **设固定显存预算**：模拟 24GB GPU、模型占 8GB、activation 等占 4GB → KV cache 预算 12GB
2. **场景 A — engine_v1（连续 KV cache）**：
   - 每 sequence 必须预分配 `max_seq_len` 显存（= prompt_len + max_tokens）
   - 即使实际只生成 32 个 token、也必须按 max_tokens=256 预算
   - 算 max_concurrent_requests = 显存预算 / per_seq_max_kv
3. **场景 B — engine_v2（paged）**：
   - 每 sequence 按当前实际长度分配 block；写满一个 block 才申请下一个
   - max_tokens=256 但实际生成 32 → 只用 2 个 block（32/16）而不是 16 个
   - 算实际占用：sum 所有活跃 sequence 实际 block 数 × block 大小
4. **跑同一 1k workload**（max_tokens=256 但实际平均生成 ~80）：
   - engine_v1 max_concurrent = ?（被 max_tokens=256 占满 limit）
   - engine_v2 max_concurrent = ?（按实际生成长度按需分配、能塞更多）
5. **写 mem_efficiency.md**：
   - 显存预算 + 每 seq 计算
   - max_concurrent 对比（v2 应该 ≥ 1.5× v1）
   - 一段解释：fragmentation（v1 每 seq 内部浪费）+ over-provisioning（v1 按 max_tokens 预算而不是实际）

### 成功标准
- max_concurrent v2 ≥ **1.5×** v1
- 文档解释清楚两种浪费来源
- 不要求 throughput 数字（v2 单 op 慢是预期）

### 失败排查
- **max_concurrent v2 < 1.5× v1**：max_tokens 和实际生成的差距不够大；调 workload 让差距更明显（如 max_tokens=512、实际平均 50）
- **v2 也 OOM**：block 释放有 bug；检查 finished request 是否真的 free 了 block

### Deliverable
- `course-b/m2-paged/mem_efficiency.md`

---

## 任务 3（选做）· Triton paged attention kernel

### 为什么做这个
你的 PyTorch 版 paged attention 慢，是因为每一步都要把零散的 block 搬回成一块连续 tensor 再算——这个"搬运"开销很大。vLLM 为什么不慢？因为它写了 CUDA kernel，直接在零散的 block 上算 attention，在 kernel 内部解引用指针，根本不搬数据。这个选做就是让你用 Triton（比裸 CUDA 友好得多）亲手体会这件事：把 indirect 访问放进 kernel 里做，跑出比 PyTorch 版快的结果。哪怕追不上 vLLM 的手写 CUDA，方向对了你就明白了"为什么工业引擎一定要下沉到 kernel 层"。

### 目标
体感"为什么 vLLM 写 CUDA kernel"——把 indirect block access 在 kernel 内做、不需要把 block gather 回连续 tensor。

### Sub-tasks
1. 装 `pip install triton`（DLAMI 一般已装）
2. 用 Triton 写一个 `paged_attention_kernel(q, key_cache, value_cache, block_table, seq_lens) -> out`
3. Triton 教程参考：https://triton-lang.org/main/getting-started/tutorials/index.html （从 vector add 到 fused softmax 入门）
4. 不要求性能追上 vLLM——只要正确性和原版 PyTorch 一致 + 比 PyTorch 版本快

### 成功标准（选做、不进过关）
- Triton kernel 输出与 PyTorch paged_attention 数值一致（atol=1e-3）
- Triton 比 PyTorch 快 ≥ 2×（仍比 vLLM CUDA 慢，但收益方向对了）

### Deliverable
- `course-b/m2-paged/engine_v2/triton_paged_attn.py`
- 一段 README："Triton 让 paged attention 快了多少 / 为什么没追上 vLLM"

---

## 三个任务做完之后

- 跑 QUIZ.md
- B-M2 过关 → 开 B-M3（Tensor parallel + 多机部署）
