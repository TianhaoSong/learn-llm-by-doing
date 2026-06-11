# C-M1 · Doing 任务详细规格

> Topic：Multi-agent 拆分判定 + Handoff 设计 + 错误处理升级路径。**只写架构决策文档**——载体是该 multi-agent 项目（API 文档 → DAG）。
>
> 三个文档：`agent_topology.md` / `handoff_scenarios.md` / `split_decisions.md`，都放在项目 `docs/architecture/`。

---

## 任务 1 · `agent_topology.md` — Agent 拆分图

### 为什么做这个
一个 multi-agent 系统跑起来后，"到底有几个 agent、谁能看到什么、谁把活交给谁"这些事很容易停留在脑子里，没人能说清。把它画成一张图、每个 agent 写明职责和上下文边界，你才会发现哪些 agent 其实在做重复的事、哪个 agent 偷看了它不该看的上下文。这个练习不是写代码，是逼你把含糊的架构直觉变成能被别人审查的明确决策。

### 目标
画清楚当前 multi-agent 系统：每个 agent 的职责 / 工具集 / 上下文边界 / 输入输出。

### 必填结构
1. **总览图**（mermaid 或 ASCII art）：每个 agent 是节点、handoff 是边、标注边的方向 + 触发条件
2. **Per-agent spec**（每个 agent 一段）：
   - **Name**：agent 名称
   - **Role**：一句话职责（如"解析 API doc 抽取 endpoint"）
   - **Tools**：能用的工具列表（read_doc / parse_openapi / call_llm 等）
   - **Context boundary**：能看到什么（来源 = handoff payload + 自己 query 的工具结果）；不能看到什么（其他 agent 的 thought）
   - **Inputs**：上游 agent handoff 给它的 schema（指向 `handoff_scenarios.md`）
   - **Outputs**：handoff 给下游或返回给 supervisor 的 schema
3. **Supervisor / orchestrator**（如有）：单独一段说明它的职责（route / retry / escalate）

### Code review checkpoint
- 每个 agent 都有 explicit 的 context boundary（不写"它会看所有需要的"，要列具体）
- 工具集写出来——多个 agent 共享同一 tool 是 OK 的，但要标"为什么不合并这两个 agent"

### Deliverable
- 在项目 `docs/architecture/agent_topology.md`
- 公开版（在 learn-llm-by-doing repo 里放一份脱敏版本作为简历参考）：`course-c/m1-architecture/agent_topology_redacted.md`

---

## 任务 2 · `handoff_scenarios.md` — 5 个真实场景

### 为什么做这个
multi-agent 最容易坏的地方就是 agent 之间交接的那一刻：A 干完活把结果丢给 B，如果交接的是一段随便写的自然语言，B 经常会理解错、或者拿到残缺的信息接着往下错。逼自己把 5 个真实交接场景写成结构化 schema（包括"对方拒绝怎么办、超时谁负责重试"），你才会发现自己的 handoff 设计有多少漏洞。写不出 5 个真实场景，本身就是个信号——说明这系统可能根本没必要拆成多个 agent。

### 目标
列 5 个真实 handoff 场景；每个写 (sender state, message schema, receiver expectation)。这是拷问 handoff 设计的核心证据。

### 必填结构（每个场景一段）
1. **场景名**（如"endpoint 解析完成 → DAG 生成"）
2. **Sender**：是哪个 agent、它处于什么状态（"已完成 N endpoint 抽取，cost X token"）
3. **Trigger**：什么条件触发 handoff（"endpoint 数 ≥ 1 且 schema_validation 通过"）
4. **Message schema**（必须用结构化 schema！）：
   ```python
   class EndpointHandoffPayload(BaseModel):
       task_id: str
       parsed_endpoints: list[Endpoint]
       confidence: float
       next_step: Literal["validate", "generate_dag"]
       sender_trace_id: str  # 用于回溯
   ```
5. **Receiver expectation**：接收方拿到 payload 后做什么（"validate endpoint 完整性 → 不通过则拒绝、handoff 回 sender"）
6. **Failure modes**：
   - sender 给了 invalid schema？
   - receiver 拒绝？谁重试？多少次？
   - 超时？默认怎么做？

### 5 个场景至少覆盖
- (a) 正常成功 handoff
- (b) Receiver 拒绝（schema 不完整）
- (c) 长链 handoff（A → B → C，C 又 handoff 回 A）
- (d) Parallel handoff（A 同时 handoff 给 B 和 C，两个结果汇合）
- (e) Escalation（下层处理不了升级到 supervisor）

### Code review checkpoint
- **每个 handoff 都有 schema**——不允许出现 "传一段自然语言 string"
- schema 用 Pydantic 或 dataclass 写出来、不是用 dict + 注释
- 失败模式写明确"谁负责重试"——不能模糊

### 失败排查
- **写不出 5 个真实场景**：说明项目其实没那么 multi-agent；或者 agent 拆得太粗（实际只有 1 个大 agent）；这是 split_decisions.md 要回答的——是不是该重新拆分

### Deliverable
- `docs/architecture/handoff_scenarios.md`（项目内）
- 脱敏版：`course-c/m1-architecture/handoff_scenarios_redacted.md`

---

## 任务 3 · `split_decisions.md` — 拆 agent 决策记录

### 为什么做这个
很多人一上来就把系统拆成好几个 agent，觉得这样"更专业"，但 multi-agent 是有实打实代价的——更慢、更烧 token、更难 debug。这个练习逼你对每一个拆分点回答"为什么不合并成一个"，并且诚实承认其中至少有一处其实该合并。能讲清楚"我哪里拆、哪里不拆、为什么"，比盲目堆 agent 更能证明你真的理解架构权衡。

### 目标
对 topology 里每个拆分点，写"为什么这里要拆而不是合并"。这是面试拷问"你为什么拆成 N 个 agent 而不是 1 个"的核心答案。

### 拆 agent 的 5 项判定标准
对每个拆分点，至少满足其中一条：
1. **上下文超 100k**——超不过的合在一起更简单
2. **工具集完全不重叠**——读 doc 的 agent 不需要写 code 工具
3. **需要不同 system prompt 风格**——research agent 要 "explore broadly"、validation agent 要 "be strict"
4. **需要独立 eval**——希望分别量化"endpoint 抽取准确率"和"DAG 生成准确率"
5. **没必要拆就不拆**——multi-agent 是成本（latency / token / debug 复杂度）

### 必填结构
对每个拆分点（即 topology 里的每个 agent 边界），写：
1. **上下两个 agent 名**
2. **判定标准命中哪几条**（≥ 1 条具体理由）
3. **如果合并会发生什么**（具体推演："合并后 system prompt 会有冲突；exploration vs validation 的指引互斥"）
4. **代价对比**：拆 = +X latency / +Y token cost / +Z debug 复杂度；不拆 = ...
5. **决策**：保持拆分 / 该合并 / 待定

### Code review checkpoint
- 至少有一个拆分点的判定结论是"该合并"——不要全 rationalize 当前架构；如果检视后发现都该拆，**那就特意标注**这个反思过程（学习态度比结论重要）

### Deliverable
- `docs/architecture/split_decisions.md`（项目内）
- 脱敏版：`course-c/m1-architecture/split_decisions_redacted.md`

---

## 三个文档做完之后

- 跑 QUIZ.md
- C-M1 过关 → 开 C-M2（Agent eval）
