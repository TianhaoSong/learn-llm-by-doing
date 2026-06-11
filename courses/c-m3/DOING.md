# C-M3 · Doing 任务详细规格

> Topic：公开 benchmark 上严格控制变量跑 single vs multi、给三元组 (cost, latency, success rate)。简历版加菜、走自己时间。
>
> 三个任务：选 task / 实现两个 agent / 实验报告。

### 任务选择
**推荐 SWE-bench Lite 子集（30-50 个 task）**——金标准、有 docker-based eval；嫌重选 HumanEval / MBPP / 自定义 codegen。核心要求：**任务足够开放、可量化判分**。

---

## 任务 1 · 选 task + 准备数据

### 为什么做这个
要想用真实数据说"single agent 还是 multi-agent 更好"，得先有一批公开、可量化判分的任务，否则结论就只是个人感觉。选对 benchmark 很关键：任务要足够开放（不然分不出架构差异）、又要能自动判对错（不然没法跑几十条算成功率）。另外公开 benchmark 可能被训练数据污染过，提前把这个风险写清楚，免得最后数字好看却经不起追问。

### Sub-tasks
1. 选 benchmark：
   - **SWE-bench Lite**：300 个 task、有 docker eval；选 30-50 跑 demo
   - **HumanEval**：164 个 codegen task、轻量 unittest eval
   - **自定义 codegen**：自己设计 30 个 task + reference solution + test cases；适合"想避免 eval 泄漏"
2. 数据下载：`pip install swebench` 或 HuggingFace `princeton-nlp/SWE-bench_Lite`
3. **eval 泄漏 disclaimer**：公开 benchmark 训练数据可能 contaminate；写在 README

### Deliverable
- `course-c/m3-comparison/dataset/` （benchmark 子集）
- `course-c/m3-comparison/data_disclaimer.md`：用了什么 benchmark、可能 contamination 风险

---

## 任务 2 · `single_agent_baseline.py` + `multi_agent.py`

### 为什么做这个
"multi-agent 更强"这种话谁都会说，但要拿出可信的对比数据，关键在严格控制变量：同 model、同工具、同输入、同判分标准，唯一不同的只有架构本身。只要有一个变量没控住，别人就能说你的结论是噪声。亲手把两个实现的变量对齐，你才会真正体会到"做一个站得住脚的对照实验有多难"——这正是这条加菜线想给你的真实数据底气。

### 目标
两个实现严格控制变量——同 model / 同 tools / 同输入 / 同 eval；唯一变量是"single vs multi"。

### Sub-tasks

#### 2.1 `single_agent_baseline.py`
- 一个 agent + 全部工具（read_file / list_dir / run_shell / write_file / run_tests）
- 用 C-M0 的 mini_agent 模板扩展
- 主循环：直接给 task → agent loop tool use 直到 stop → 拿结果

#### 2.2 `multi_agent.py`
- 把该 multi-agent 项目（API 文档 → DAG）的拆分思路迁移到本任务，例如 codegen 拆：
  - **Planner agent**：read task → 拆成 subtask plan
  - **Coder agent**：read plan → 写代码
  - **Reviewer agent**：read code → 跑 test → 决定 accept / send back
- 同样的 tool set
- Handoff 用 schema（C-M1 的设计）

#### 2.3 控制变量 checklist
- 同 model (claude-sonnet-4-6 都用)
- 同 tools（同 schema、同 implementation）
- 同 input format
- 同 eval（task 完成 = test pass）
- 唯一区别：架构（single vs multi）

#### 2.4 Cost 监控
- 每次 API call 后累计 `input_tokens + output_tokens`
- 算成本：input × $3/MT + output × $15/MT（Claude Sonnet 4.6 价格、查最新）
- 30 个 task × ~3 万 tokens = ~$50 一轮（合理预算）

### 成功标准
- 两个实现都跑通 ≥ 80% task（不挂掉）
- Cost 监控数字给出
- 控制变量 checklist 在 README 列清楚

### 失败排查
- **single agent 一直循环不停**：max_iterations 太小或 task 太难；不要无限重试，过 N 次就标 failed
- **multi agent handoff 失败**：handoff schema 写不严格；用 Pydantic + try/except 让 sender 看到 schema validation error 自己 fix
- **cost 飙到 $200+**：可能 multi agent 死循环（C-M1 题 6 提到的 (A,B) 来回）；加回环检测

### 辅助阅读（非 canonical）
- SWE-bench README：https://github.com/princeton-nlp/SWE-bench
- Anthropic API pricing：https://www.anthropic.com/pricing
- HumanEval：https://github.com/openai/human-eval

### Deliverable
- `course-c/m3-comparison/single_agent_baseline.py`
- `course-c/m3-comparison/multi_agent.py`
- `course-c/m3-comparison/run_experiment.sh`：一键跑两个 agent on N tasks

---

## 任务 3 · `experiment_report.md`

### 为什么做这个
跑完实验只是攒了一堆数字，真正值钱的是把它们浓缩成一个有数据支撑的判断：什么样的任务该拆成 multi-agent、什么样的单 agent 加好工具就够了。这一页报告逼你给出 (成本, 延迟, 成功率) 三元组对比，并诚实标明 sample size 和泄漏等局限。哪怕结论是"single 赢了"也很好——能拿出实验数据说"multi-agent 不是银弹"，比空喊口号有说服力得多。

### 目标
≤ 1 页的报告：结论 + 一张表 + 一段 trade-off 分析。这是简历直接附的一页。

### 结构
```markdown
# Single-agent vs Multi-agent on SWE-bench Lite (N tasks)

## 实验设置
- model: claude-sonnet-4-6
- tasks: 30 from SWE-bench Lite
- tools: read_file, list_dir, run_shell, write_file, run_tests
- 唯一变量: 架构（1 agent 全工具 / 3 agent 拆分）

## 结果

| Metric | Single | Multi | Δ |
|---|---|---|---|
| Success rate | 50% | 53% | +6% |
| Avg cost / task | $0.40 | $1.20 | 3× |
| Avg latency / task | 30s | 90s | 3× |

## 分析
- Multi 赢的 case 主要在...
- Multi 输的 case 主要在 (短 task、handoff overhead 占比大、planner 拆出错误的 subtask 后 coder 一路错下去)
- 哪种 task 该拆: 长 / 复杂 / 多步骤推理
- 哪种 task 不该拆: 短 / 单步 / 工具集小

## 局限
- N=30 sample size 风险
- Eval 泄漏可能（SWE-bench 公开）
- 单一 model；其他 model 结论可能不同

## 结论
"Multi-agent is not a free win. For X type tasks, it pays off; for Y type, single-agent + good tools wins."
```

### 成功标准
- ≤ 1 页（约 500 字）
- 三元组对比表给出
- 分析部分指出"哪种 task 该拆 / 哪种不该拆"——这是面试可讲的核心
- **结果可能是 single 赢——这是好结论**；面试讲"我做了实验、发现 multi-agent 不是银弹"

### Deliverable
- `course-c/m3-comparison/experiment_report.md`

---

## 三个任务做完之后

- 跑 QUIZ.md
- C-M3 过关 → 简历版加菜 done → 选做 C-M4（Slack agent 工具设计）
