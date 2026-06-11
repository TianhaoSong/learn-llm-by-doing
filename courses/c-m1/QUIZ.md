# C-M1 · 知识自查口试

> 验证拆 agent 判定 / handoff 设计 / 错误升级路径是否吃透。30 秒内答得出 = 过。

---

## 拆 agent 判定（对应任务 3）

1. **拆 agent 的 5 项判定标准是什么？**
   - 想点：(1) 上下文超 100k 该拆；(2) 工具集完全不重叠该拆；(3) 需要不同 system prompt 风格该拆；(4) 需要独立 eval 该拆；(5) 没必要拆就不拆——multi-agent 是成本

2. **multi-agent 的成本是什么？什么场景反而不该拆？**
   - 想点：成本 = (a) latency（每次 handoff 多一次 LLM call）；(b) token 消耗（每个 agent 重复带 context）；(c) debug 复杂度（trace 散在多个 agent）；(d) handoff 失败模式多；不该拆：上下文不大、工具集重叠、不需要独立 eval、单 agent 就能 reasoning 到位

3. **如果一个任务用 1 agent 5 工具能完成、用 3 agent 各 2 工具也能完成，怎么选？**
   - 想点：1 agent 简单——优先；除非 (a) 工具集多到 model 调错率高（多 tool 时 model 容易混）、(b) 需要不同 prompt 风格、(c) 需要独立 eval；多 agent 是优化、不是默认

---

## Handoff 设计（对应任务 2）

4. **Handoff 用结构化 schema vs 裸字符串——为什么前者必选？**
   - 想点：(a) schema 强制 sender 把所有 receiver 需要的信息显式列出（不会"以为他知道"）；(b) 失败模式可枚举（schema 校验失败 = 立即重试 / 拒绝）；(c) trace 可读、debug 容易；(d) 模型调对率高（structured output）

5. **共享 memory（vector store / 全局 state）vs 显式 payload，默认选哪个？**
   - 想点：默认 **显式 payload**——耦合低、debug 容易、agent 是无状态的可独立 eval；只有 payload 大到不合理（如要传几 MB 文本）才考虑共享 memory + 引用 id

6. **回环检测怎么做？为什么需要？**
   - 想点：每次 handoff 记 (from_agent, to_agent)；连续 3 次相同对 → escalate 到 supervisor；不检测 → A↔B 死循环烧 token；常见诱因：A 给 B 的 payload 不全、B 拒绝 handoff 回 A、A 又生成同样的不全 payload

7. **Parallel handoff（A 同时给 B 和 C）的难点是什么？**
   - 想点：(a) 等齐——B 快 C 慢，A 等多久；(b) 一致性——B 和 C 的结果矛盾怎么办；(c) cost——如果只用一个就行不该 parallel；(d) cancel——B 完成后 C 还在跑要不要杀；推荐策略：明确 timeout + 多数表决 + 矛盾时升级 supervisor

---

## 错误处理与升级（对应任务 2 + 3）

8. **Handoff 失败（下游 agent 拒绝 / 超时）时谁负责重试？**
   - 想点：sender 重试有限次（如 3 次）→ 超过升级到 supervisor；不要让 receiver 自己尝试 fix（容易循环）；重试时 sender 应该改 payload（给更多 context / 改 strategy）、不是同样请求重发

9. **什么时候该升级到 supervisor agent？**
   - 想点：(a) 重试 N 次失败；(b) 检测到回环（连续 3 次同 (A, B) handoff）；(c) 下游 agent 报 "out of scope"；(d) cost / latency 超 budget；supervisor 决策：换 strategy / 换 agent / fail 给 user

10. **Tool 失败 vs handoff 失败——处理方式有什么区别？**
    - 想点：tool 失败 = agent 内部、自己处理（看错误 → 改 args / 换 tool / 报告）；handoff 失败 = 跨 agent 边界、需要 sender 决策（重试 / 换 receiver / 升级）；前者由 agent 自己 ReAct 解决、后者由 supervisor 仲裁

---

## 综合（覆盖整个 C-M1）

11. **如果面试官问"你这个任务为什么拆 N agent 而不是 1 个"——你按什么结构答？**
    - 想点：(1) 任务的核心 sub-task 列出来；(2) 哪两个 sub-task 之间的 boundary 命中了拆 agent 5 项判定（具体哪条）；(3) 如果合并会发生什么（具体推演）；(4) 代价对比（latency / token）；(5) 决策

12. **如果你的 multi-agent 系统比 single-agent 慢 3× 但效果只好 5%——值不值？**
    - 想点：看产品场景；如果是 (a) 高 stake low frequency（如生产代码生成），值；(b) 低 stake high frequency（如 chat），不值；(c) 还要看 latency 是否在用户感知阈值内（如 chat < 2s）；不要套统一答案、看场景

---

## 自查标准

- 12 题里 ≥ 10 题 30 秒内答得出 → C-M1 过关
- 题 1-3（拆判定）和 题 4-7（handoff 设计）必答好——这是面试核心
