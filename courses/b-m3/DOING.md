# B-M3 · Doing 任务详细规格

> Topic：Tensor parallel 在 attention/FFN 上的切法 + 通信模式 + NVLink/EFA/TCP 拓扑对 TP 效率的影响 + 推理 TP vs 训练 FSDP 取舍。
>
> 三个任务：`engine_v3/` 实现 TP / 单机扩展曲线 / 跨机部署 + 报告。

### 模型选择
切到 **Llama-2-7B**——TP 数字才好看；GPT-2 small 太小、切了反而慢（all-reduce 开销 > 收益）。
```bash
huggingface-cli login  # 接受 license
huggingface-cli download meta-llama/Llama-2-7b-hf
```

---

## 任务 1 · `engine_v3/` — 手写 Column / Row parallel

### 为什么做这个
一个 7B 甚至 70B 的模型单卡装不下，怎么办？tensor parallel 把每一层的权重矩阵切开、摊到多张卡上并行算。但魔鬼在细节里：切完之后什么时候必须把各卡的部分结果合起来（all-reduce）？合多了就慢、合少了就错。Megatron 的核心 trick 是——把矩阵按"列切"和"按行切"交替排，让前一层的输出正好是后一层的输入，每个 transformer layer 只需要 2 次 all-reduce。你不调 PyTorch 的现成 API、自己手写一遍 column/row parallel 的 Linear，就是为了亲眼确认这 2 次通信发生在哪、为什么不能更多也不能更少。这是理解多卡推理的命门。

### 目标
不用 PyTorch 的 `parallelize_module` API，自己写 column-parallel / row-parallel 的 Linear，理解每一次 all-reduce 在哪里发生。

### Sub-tasks

#### 1.1 Column-parallel Linear
把 `Y = X @ W` 沿 W 的列切：每 rank 持有 `W_i = W[:, i*hidden/N : (i+1)*hidden/N]`，算 `Y_i = X @ W_i`、各 rank 持有部分列结果。
```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        super().__init__()
        assert out_features % tp_size == 0
        self.weight = nn.Parameter(torch.empty(out_features // tp_size, in_features))
        # bias 也切
    def forward(self, x):
        # x: [B, T, in_features]（所有 rank 都有完整 x）
        return F.linear(x, self.weight)  # 输出 [B, T, out_features // tp_size]
        # 不做 all-reduce —— 由后续 row-parallel 做
```

#### 1.2 Row-parallel Linear
把 `Y = X @ W` 沿 W 的行切：每 rank 持有 `W_i = W[i*hidden/N : (i+1)*hidden/N, :]` 和 `X_i = X[:, :, i*hidden/N:...]`（输入也切了，正好接 column-parallel 输出），算 `Y_i = X_i @ W_i`、然后 **all-reduce** 把 N 个 partial sum 加起来。
```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        super().__init__()
        assert in_features % tp_size == 0
        self.weight = nn.Parameter(torch.empty(out_features, in_features // tp_size))
    def forward(self, x):
        # x: [B, T, in_features // tp_size]（接 column-parallel 输出）
        out = F.linear(x, self.weight)  # [B, T, out_features]
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out
```

#### 1.3 改造 attention
- QKV projection：column-parallel——把 Q/K/V 三个矩阵合并成大矩阵切；每 rank 拿 `n_head / tp_size` 个 head；这是为什么 `n_head` 必须能被 `tp_size` 整除
- attention 本身（Q @ K.T → softmax → @ V）：每 rank 独立算自己那部分 head（无通信）
- output projection：row-parallel——这一层会触发 all-reduce 把所有 rank 的 partial output 合并

#### 1.4 改造 FFN
- 第一个 linear（n_embd → 4×n_embd）：column-parallel
- GELU：每 rank 独立算
- 第二个 linear（4×n_embd → n_embd）：row-parallel；触发 all-reduce

#### 1.5 关键观察
- 每 transformer layer 只有 **2 次 all-reduce**：attention output 一次、FFN output 一次
- 这是 Megatron 的核心 trick——column-parallel 的输出**正好**是 row-parallel 的输入，中间不需要通信

#### 1.6 集成到 engine
- 改造 B-M2 的 `engine_v2` → `engine_v3`：替换 Linear、跑 `torchrun --nproc_per_node=N -m engine_v3.engine`
- KV cache 也按 head 维切（每 rank 只存自己那部分 head 的 KV）
- generation 主循环逻辑不变

### 成功标准
- TP=2 / TP=4 单机能跑通 Llama-2-7B
- 输出与 TP=1（单卡）的 baseline 逐字一致（greedy 验证）
- Code review checkpoint：
  - 每层只有 2 次 all-reduce（attention out + ffn out）；多了说明实现错了
  - QKV 合并成单一 column-parallel matmul（而不是 3 个独立的 ColumnParallelLinear）

### 失败排查
- **TP=2 输出和 TP=1 不一致**：QKV 切的方式不对（应该按 head 切而不是按维度切）；或者 row-parallel 的输入没正确从 column-parallel 拿（中间多了一次 all-gather）
- **每层 4 次 all-reduce**：attention 和 ffn 内部各做了一次额外 all-reduce；不需要——column→row 自带的 all-reduce 已经足够
- **跑得慢**：检查 tp_size 是否整除 n_head（Llama-2-7B 是 32 head，TP=2/4/8 都行；TP=3/5 不行）

### 辅助阅读（非 canonical）
- vLLM `parallel_state.py`（参考工业实现）：https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py
- PyTorch TP API（`parallelize_module`，知道工业 API 长什么样）：https://pytorch.org/docs/stable/distributed.tensor.parallel.html

### Deliverable
- `course-b/m3-tp/engine_v3/{linear_parallel.py, model_tp.py, engine.py}`
- `tests/test_tp_correctness.py`：TP=4 vs TP=1 输出逐字一致

---

## 任务 2 · 单机 TP 扩展曲线

### 为什么做这个
TP 把模型切到 8 张卡，是不是就能快 8 倍？现实里几乎不可能——因为每层那 2 次 all-reduce 占在关键路径上，没法和计算完全重叠，卡越多通信占比越大。这一步就是把 TP=1/2/4/8 都跑一遍、画出扩展曲线，亲眼看到 efficiency 是怎么从 100% 一路掉下来的，并能解释损失来自哪里。偶尔还会撞见 super-linear（看起来超过线性加速）——但那其实是单卡装不下被迫压小 batch 带来的显存收益，不是 TP 本身的功劳。能分清这两种收益，才算真懂了多卡扩展的账。

### 目标
在单机 8× A100 NVLink 上跑 TP=1/2/4/8，给出 (TTFT, TPOT, throughput) 三元组、画 scaling 曲线。

### Sub-tasks
1. 准备 fixed workload：100 个 prompt（长度均匀分布）、max_tokens=128、greedy
2. 跑 TP=1/2/4/8（TP=1 是单卡 baseline）
3. 每组记 5 次 median：TTFT_p50, TPOT_p50, throughput
4. 画 scaling efficiency 曲线：`speedup(TP=N) / N` vs N
5. 标注 super-linear / sub-linear：
   - **Sub-linear** 是常见情况——TP 加倍但 all-reduce 时间不能完全和 compute overlap
   - **Super-linear** 偶尔出现——单卡 OOM（KV cache 装不下）→ batch 被压小、TP 后能放大 batch；这是显存收益不是 TP 收益本身

### 成功标准
- TP=8 efficiency ≥ **70%**（Llama-2-7B + 8× A100 NVLink）
- 解释 efficiency < 1 来自哪里——主要是 all-reduce 没完全 overlap

### 失败排查
- **TP=4 比 TP=2 慢**：模型还不够大、all-reduce overhead 占比大；切到 13B 模型就好
- **TP=8 efficiency < 50%**：检查互联——`nvidia-smi topo -m` 看是否真的 NVLink；如果是 PCIe（消费卡）那就是预期

### Deliverable
- `course-b/m3-tp/tp_scaling_single_node.md`：硬件 + 表 + 曲线 + 解释

---

## 任务 3 · 跨机 TP（📖 默认读懂，拿到 ≥2 节点才动手）

### 为什么做这个
单机 8 卡装不下的超大模型（比如 70B），就得跨机器做 TP。但这里有个反直觉的结论你必须懂：跨机几乎总是慢一个数量级。原因是 TP 的 all-reduce 在关键路径上、每个 token 都要等它，而机器之间的网络（EFA ~100 GB/s，配错还会掉到 TCP ~10 GB/s）比单机内的 NVLink（~600 GB/s）慢一个数量级。所以单机 TP=8 通常比跨机 TP=16 还快——除非模型大到一台机器真的塞不下。这个权衡读懂就能讲清、也能在面试和 QUIZ 里答清，不用真搭 EFA 集群（那个配置坑多、成本高）才算掌握，所以默认只读懂、有多节点资源再当 bonus 动手。

> **降级理由**：跨机 TP 的核心知识是「互联带宽差一个数量级 → 跨机 all-reduce 成关键路径瓶颈」，这个**读懂就能讲清、也能在 QUIZ 答清**。真跑跨机要 ≥2 节点 + EFA placement group + onboarding，成本高、配置坑多，**不是 B-M3 的必需动手项**。默认读懂；只有你恰好拿到多节点资源、想验证 EFA，才当 bonus 动手。

### 📖 读懂部分（QUIZ 自查，不需动手）
- **互联带宽三档**：NVLink/NVSwitch ~600 GB/s（同节点）> EFA ~100 GB/s（跨节点高速网）> TCP ~10 GB/s（fallback）。**差一个数量级**。
- **为什么跨机 TP 慢一个数量级**：TP 每层有 2 次 all-reduce（attention output + FFN output）在**关键路径**上——每个 token 都要等这些 all-reduce 完成。跨机时这些 all-reduce 走 EFA（甚至 TCP），比单机 NVLink 慢一个数量级 → 整体 latency 慢一个数量级。
- **关键路径是哪些 op**：attention output projection（row-parallel）的 all-reduce + FFN 第二个 linear（row-parallel）的 all-reduce。
- **EFA 配错会退化 TCP**：cluster placement group + AMI EFA driver + IAM + `NCCL_PROTO=Simple`/`FI_PROVIDER=efa`，任一错 → 掉到 TCP、带宽再掉 10×。这是为什么跨机部署容易踩坑。
- **结论**：单机 TP=8（NVLink）几乎总比跨机 TP=16（EFA）快——除非模型大到单节点 8 卡装不下（如 70B），才不得不跨机。

QUIZ 自查：「推理 TP=8 跨机比单机慢多少？为什么？哪些 op 是关键路径？」

### 🔧 Bonus（仅当你拿到 ≥2 节点 + EFA）
真跑跨机验证上面的数：
1. 起 2 节点 p4d + cluster placement group + EFA（`NCCL_PROTO=Simple` `FI_PROVIDER=efa`）
2. 先用 A-M2 `bench_collectives.py` 跨节点版验证 EFA 生效（all-reduce 1GB 应 ~80-100 GB/s；只有 10 GB/s = TCP fallback）
3. `torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$NODE_RANK --master_addr=$MASTER_IP ...` 跑 TP=16，对比单机 TP=8 的 (TTFT, TPOT)
- 参考：[AWS EFA 配置](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) / [NCCL multi-node](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)

### 成功标准
- 默认：QUIZ 能答清「跨机为什么慢一个数量级 + 关键路径是哪些 op」= 过关
- Bonus（如跑了跨机）：给出 single-node TP=8 vs cross-node TP=16 的 latency 对比 + EFA 是否生效

### Deliverable
- `course-b/m3-tp/tp_cross_node.md`：配置 + 数字 + 一段分析

---

## 三个任务做完之后

- 跑 QUIZ.md
- B-M3 过关 → 开 B-M4（与 vLLM/SGLang 对比）
