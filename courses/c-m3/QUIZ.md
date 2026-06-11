# C-M3 · 知识自查口试

> 验证公平对比 / multi-agent 输了的归因 / cost-benefit 决策是否吃透。30 秒内答得出 = 过。

---

## 公平对比（对应任务 2.3）

1. **Single vs multi agent 公平对比的控制变量是什么？**
   - 想点：(1) 同 model；(2) 同 tools（schema + 实现）；(3) 同 input format；(4) 同 eval；唯一变量是架构（single agent / multi agent）

2. **如果你 multi-agent 用了 GPT-4 而 single 用了 Claude——结果可信吗？**
   - 想点：不可信——结果差异可能来自 model 本身、不是架构；公平对比必须同 model

3. **三元组 (cost, latency, success rate) 缺一不可——为什么不能只比 success rate？**
   - 想点：多 agent 通常 success rate 高一点点 + cost 3× + latency 3×；只比 success rate 看起来纯赢、忽略代价；产品决策要看 cost / latency 是否在阈值内

---

## Multi-agent 输了的归因（对应任务 3）

4. **如果 multi-agent success rate 输给 single——可能的原因有哪几类？**
   - 想点：(1) 拆分本身错了（agent 边界不合理、handoff 信息损失）；(2) 哪个 agent 是瓶颈（planner 拆错 subtask → coder 一路错）；(3) handoff overhead 在短 task 上占比大；(4) prompt 没调好（每个 agent 的 prompt 都独立、需要单独优化）

5. **怎么定位是哪个 agent 是瓶颈？**
   - 想点：(1) trajectory eval 看每个 agent 的 step 合理率；(2) 失败 case 复盘 trace、看从哪一步开始走偏；(3) ablation：去掉某个 agent 用 single 替换、看结果变化

6. **如果 multi-agent 在长复杂 task 上赢、短简单 task 上输——结论是什么？**
   - 想点：multi-agent 是 tooled-up overhead——overhead 摊到长 task 上 ROI 高、短 task 上 ROI 低；选用决策应该按 task 复杂度分流；这是面试讲故事的核心

---

## Eval 泄漏（对应任务 1）

7. **公开 benchmark（SWE-bench / HumanEval）有 contamination 风险——为什么？怎么 disclaimer？**
   - 想点：model 训练数据可能包含 benchmark；模型见过题目 → 不公平；disclaimer：在 README 注明用了什么 benchmark、可能 contamination；理想做法是搭配自定义 holdout 任务对照

8. **怎么减少 contamination 影响？**
   - 想点：(1) 自定义任务作 holdout（你的该项目本身就是）；(2) 公开 benchmark 加扰动（修改 prompt / 测例）；(3) 看相对差距（multi vs single 的差不会因为 contamination 变化）

---

## Cost / Benefit 决策（对应任务 3）

9. **multi-agent 比 single 慢 3× / 贵 3× / 但 success rate 高 5%——值不值？**
   - 想点：看场景；(a) 高 stake low frequency（生产代码、医疗）→ 值；(b) 低 stake high frequency（chat、推荐）→ 不值；(c) 看 latency 是否在用户感知阈值内（chat < 2s 是死线）；不要套统一答案

10. **如果只能再优化一项 single 或 multi，你优化哪个？为什么？**
    - 想点：先看哪个有更大优化空间——success rate 差距小（< 5%）但 cost / latency 差距大 → 优化 multi 的 overhead（handoff overhead / scheduling）；success rate 差距大 → 优化 single 的工具集 / prompt

---

## Sample Size & 统计（对应任务 1 + 3）

11. **N=30 task 的 success rate 80% 给的 95% CI 大概多宽？这意味着什么？**
    - 想点：±14% 量级（Wilson interval）；意思是 multi 比 single 高 5% 完全可能是噪声；面试讲故事 N=30 够、要主动说"我意识到 sample 风险、production 部署会扩到 ≥ 200"

12. **如果你只能跑 N=30，怎么提高结果可信度？**
    - 想点：(1) 多次实验（每个 task 跑 3 次取多数）→ 减少 model 随机性；(2) by_tag 分析（看简单 / 复杂 task 各自结论）；(3) 指明置信区间；(4) 不下绝对结论——讲"我看到 X 趋势、还需要更多数据"

---

## 综合（覆盖整个 C-M3）

13. **面试官问"这个实验的 cost 大概多少、值不值"——你怎么答？**
    - 想点：cost ~$50（30 task × 3 万 tokens × $15/MT output）；值 = 我学到了 (a) 公平对比的控制变量；(b) multi 不是银弹的实证；(c) 哪种 task 该拆的判断；这是简历直接附的一页

14. **如果你的实验结论是 single 全面赢——这对该 multi-agent 项目（API 文档 → DAG）有什么 implication？**
    - 想点：(a) 检视当前架构是不是过度拆分；(b) 该项目任务复杂度是不是和实验任务可比；(c) 不一定 1:1 迁移、但是个 prior，应该重新审视 split_decisions.md（C-M1）；学习闭环：实验 → 影响架构决策 → 改进 → 重做实验

---

## 自查标准

- 14 题里 ≥ 11 题 30 秒内答得出 → C-M3 过关
- 题 4-6（multi 输了的归因）和 题 9-10（cost-benefit）必答好
