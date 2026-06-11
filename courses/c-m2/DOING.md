# C-M2 · Doing 任务详细规格

> Topic：Agent eval 体系——end-to-end（DAG IoU）+ trajectory（每步合理性，LLM-as-judge）+ 反偏置手段。
>
> 三个任务：标 eval set / 实现 harness / 跑 baseline 写报告。

---

## 任务 1 · 标 ≥ 20 条 eval 数据集

### 为什么做这个
你没法改进一个你测不准的东西。想知道 agent 到底行不行、改了 prompt 是变好还是变坏，前提是手里有一批"标准答案"——给定输入，期望输出长什么样。这步又累又枯燥（手工标几十条想吐），但这是整个 eval 的地基：没有 ground truth，后面所有的成功率数字都是自欺欺人。

### 目标
人工标 20 条 (API doc snippet, expected DAG) 样例。这是 eval 的 ground truth。

### Sub-tasks
1. **格式**：`eval_dataset.jsonl`，每行：
   ```json
   {
     "id": "case_001",
     "input": {"api_doc_text": "..."},
     "expected_dag": {
       "nodes": [{"id": "extract_users", "task_type": "api_call", "endpoint": "/users"}, ...],
       "edges": [{"from": "extract_users", "to": "filter_active"}, ...]
     },
     "tags": ["simple", "single-endpoint"],
     "notes": "human label, version 1, 2026-06-01"
   }
   ```
2. **覆盖度**：20 条至少包含
   - 5 条简单（单 endpoint、≤ 3 节点）
   - 5 条中等（多 endpoint、有依赖）
   - 5 条复杂（join / branching / 错误 handling）
   - 5 条 edge case（malformed doc / 矛盾描述 / 空 doc）
3. **note 字段**：标"为什么这是期望输出"——给将来回看自己用，不是给 model

### 成功标准
- 20 条标完
- tags 让你能跑分类指标（simple / medium / complex / edge_case 各自的成功率）
- 这一步可能耗 ~1 整天——眼睛标完想吐就标对了

### Deliverable
- `course-c/m2-eval/eval_dataset.jsonl`（脱敏版可放公开 repo；含真实 doc 数据的放项目内）

---

## 任务 2 · End-to-end + Trajectory harness

### 为什么做这个
评 agent 有两个层面：最终结果对不对（end-to-end），和它一路走过来每一步合不合理（trajectory）。只看结果你会漏掉"瞎蒙对了"的情况，只看过程又抓不住"步步合理但答案是错的"。这两套 harness 让你同时看这两面。而且用 LLM 当裁判会引入它自己的偏见（偏爱长答案、偏爱自家 model），亲手做一遍反偏置才会明白为什么不能无脑信任 LLM-as-judge 的打分。

### 目标
两套 harness——end-to-end 算 DAG 结构匹配；trajectory 用 LLM-as-judge 评每步决策。

### Sub-tasks

#### 2.1 End-to-end harness — DAG 结构匹配（IoU 加权）
- 写 `harness_e2e.py`：
  ```python
  def normalize_dag(dag):
      # 排序 nodes by id（让 dict 顺序无关）
      # 忽略不影响语义的 attr（如 timestamp / debug_info）
      ...
  
  def dag_iou(generated, expected):
      gen = normalize_dag(generated)
      exp = normalize_dag(expected)
      node_iou = jaccard(set(n["id"] for n in gen["nodes"]), set(n["id"] for n in exp["nodes"]))
      edge_iou = jaccard(set((e["from"], e["to"]) for e in gen["edges"]), set((e["from"], e["to"]) for e in exp["edges"]))
      return 0.5 * node_iou + 0.5 * edge_iou
  ```
- 跑 20 条：每条调 agent 生成 DAG → 算 iou → 二值化（如 iou ≥ 0.8 算成功）→ 算 success rate
- 报告：(overall, by_tag) success rate

#### 2.2 Trajectory harness — LLM-as-judge
- 用 C-M0 的 trace logger 输出的 jsonl
- 写 `harness_trajectory.py`：每个 trace 的每个 step（model_thought / tool_call / tool_result）调 judge LLM 评分：
  ```
  prompt 模板:
    Given the task: {task}
    Given the previous steps: {context}
    The agent's next decision: {step}
    
    Is this decision reasonable? Answer YES or NO with one sentence reasoning.
    Ignore length differences, focus only on correctness.
  ```
- 算每条 case 的 trajectory_score = 合理 step / 总 step
- 整体 trajectory 合理率 = 平均

#### 2.3 反偏置手段
- **Position bias**：同一 (gen_a, gen_b) pair 让 judge 评两次、交换顺序、取一致；不一致 = judge 不可信
- **Length bias**：prompt 里明确 "ignore length, focus on correctness"
- **Self-preference**：judge 用不同 model（如被评估的是 Claude → judge 用 GPT-4 或反之）；至少不要让被评估和 judge 是同一 vendor 同一 family

### 成功标准
- 两套 harness 都能跑、产出数字
- IoU 计算和 normalize 写出来过单测（同一 DAG 不同节点顺序输入应该 iou=1）
- Trajectory harness 至少做了 position bias check（每 pair 两次）

### 失败排查
- **DAG IoU 永远很低**：normalize 没做对——节点 id 命名不一致（agent 生成 `fetch_users`、expected 是 `extract_users`）；用 fuzzy match 或要求 agent follow 命名规范
- **judge 一致性差**：position bias check 显示 50% 翻转——换 judge model / prompt 写得更精确

### Deliverable
- `course-c/m2-eval/harness_e2e.py`
- `course-c/m2-eval/harness_trajectory.py`
- `course-c/m2-eval/tests/test_iou.py`

---

## 任务 3 · 跑 baseline + 写 `eval_v0.md`

### 为什么做这个
有了数据集和 harness，得先量出"现在到底是什么水平"，这个数字就是基线。以后任何优化都要跟它比——没有基线，你说"我把它改好了"就是空话。同样重要的是诚实写下这个基线的局限（20 条样本只能抓明显 bug、抓不住长尾），养成"先说清楚数据能说明什么、不能说明什么"的习惯。

### 目标
在 20 条 eval set 上跑该项目 agent，给出 baseline 数字。这是后续优化的对照基线。

### Sub-tasks
1. 跑 e2e harness：20 条 case 的成功率
2. 跑 trajectory harness：20 条 case 的合理率
3. 写 `eval_v0.md`：
   - eval set 描述（20 条、tags 分布、谁标的、什么时候标的）
   - baseline 数字：
     - overall e2e success rate
     - by_tag e2e success rate（simple / medium / complex / edge_case）
     - overall trajectory 合理率
     - 失败 case 列表（哪几条挂了、哪类居多）
   - **Sample size 风险声明**：明确写"20 条只能发现明显 bug、抓不住 long-tail；production 部署前扩到 ≥ 200 条"

### 成功标准
- 数字给全（overall + by tag + 失败 case）
- 有 sample size 风险声明
- 失败 case 至少分类一次（哪几类问题最多——是 prompt 不到位 / 工具不够 / handoff 设计有 bug）

### Deliverable
- `course-c/m2-eval/eval_v0.md`

---

## 三个任务做完之后

- 跑 QUIZ.md
- C-M2 过关 → 开 C-M3（single vs multi 对照实验）
