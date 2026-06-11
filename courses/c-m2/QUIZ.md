# C-M2 · 知识自查口试

> 验证 e2e vs trajectory eval / LLM-as-judge 偏置 / sample size 风险是否吃透。30 秒内答得出 = 过。

---

## E2E vs Trajectory（对应任务 2）

1. **End-to-end eval 和 trajectory eval 各看什么？为什么需要两个？**
   - 想点：e2e = 任务最终结果是否对（DAG 结构匹配率）；trajectory = 每步决策是否合理；e2e 高 + trajectory 低 = "蒙对了"，不稳；e2e 低 + trajectory 高 = 决策合理但能力不够；都看才能判断改进方向

2. **如果 e2e 高但 trajectory 低，说明什么？**
   - 想点：agent 走了奇怪路径但巧合答对——下次容易翻车；要看哪几步不合理（prompt 误导？工具用错？）；改完后 e2e 不一定升（巧合也消失了）但 trajectory 一定升

3. **如果 trajectory 高但 e2e 低，说明什么？**
   - 想点：每步决策都合理、但拼起来不到位——可能是基础能力不够（model 太弱）、可能是工具集不全、可能是 task 本身 model 解不了；换大 model 是首选

4. **DAG 结构匹配怎么算？为什么先 normalize 再 IoU？**
   - 想点：normalize = 排序 nodes by id、忽略无关属性（debug 字段 / timestamp）；不 normalize → 同 DAG 不同顺序 IoU 不为 1；IoU = 节点集合 IoU 和边集合 IoU 加权（如 0.5 + 0.5）

---

## LLM-as-Judge 偏置（对应任务 2.3）

5. **LLM-as-judge 的 3 大偏置是什么？怎么反？**
   - 想点：(1) **Position bias** = judge 偏好放第一个的——同 pair 跑两次交换顺序取一致；(2) **Length bias** = judge 偏好长输出——prompt 里 "ignore length, focus on correctness"；(3) **Self-preference** = judge 偏好自家 model 的输出风格——judge 用不同 vendor model

6. **为什么 judge 不要和被评估用同一 model 同一 family？**
   - 想点：self-preference——被评估是 Claude → 让 GPT-4 或 Gemini 当 judge；同一 model 当 judge 时偏好"和自己风格相似"的输出（即使另一个更对）；不同家 model 互评更可信

7. **Position bias check 的具体做法？**
   - 想点：每 pair (a, b)，跑两次：judge_score(a, b) 和 judge_score(b, a)；如果两次结果一致（都 a 赢 或 都 b 赢） → 可信；不一致 → 把这一对标 "uncertain" / 增加人工抽样

---

## Sample Size 风险（对应任务 3）

8. **eval 集 20 条会不会太少？什么情况下要扩到 200 条？**
   - 想点：20 条能发现"完全跑不通"的明显 bug、抓不住 long-tail（成功率 80% 的 95% CI 大概是 ±18% — 噪声很大）；production 部署前必须扩到 ≥ 200 条；面试讲故事 20 条够、但要主动说"我意识到 sample size 风险"

9. **如何选 eval set 的 tag 分布？**
   - 想点：(1) 覆盖真实场景中的 case 分布（按真实数据分布估比例）；(2) 故意 oversample 难 case（edge case 占 20-30%）让 eval signal 显著；(3) 至少 3 类 tag 让你能跑 by_tag 分析

---

## Online / Shadow Eval（覆盖学习目标但任务没展开）

10. **Offline / Online / Shadow eval 各是什么？**
    - 想点：Offline = 离线 eval set（你的 20 条）；Online = 真实流量上跑 + 收集 user feedback / 业务指标；Shadow = 真实流量上跑但不影响用户（替身、对比 prod 输出）；Shadow 是部署前最后一道关

11. **Online eval 比 offline 强在哪？为什么不能只靠 online？**
    - 想点：online = 真实分布 + 大样本 + 业务指标；但 (a) 慢（要跑流量）；(b) 不能跑实验性变更（伤用户）；(c) 反馈延迟（用户 1 周后才回来）；offline 是开发期快速迭代必要的

---

## 综合（覆盖整个 C-M2）

12. **如果你的 e2e 成功率从 60% 升到 70%，但 trajectory 合理率从 80% 降到 70%——这是好还是坏？**
    - 想点：可疑——agent 找到了"高效但不合理"的捷径，可能是 prompt hack 或 reward gaming；e2e 升的可能是单一 case 类型（看 by_tag），trajectory 降说明可解释性差了；建议回退、找 trajectory 升 + e2e 升的改进

13. **面试官问"你 eval 怎么防止 model 在 eval set 上过拟合"——你怎么答？**
    - 想点：(1) eval set 不进 training（基础）；(2) 定期换 fresh holdout（模型迭代时）；(3) 用 contamination check（看 model 是否见过 prompt）；(4) 关键看 online + shadow 数据、不只 offline；(5) eval 集本身公开 → 必须 disclaimer

---

## 自查标准

- 13 题里 ≥ 11 题 30 秒内答得出 → C-M2 过关
- 题 1-3（e2e vs trajectory）和 题 5-7（judge 偏置）必答好
