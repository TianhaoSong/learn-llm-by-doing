# C-M0 · 知识自查口试

> 验证 ReAct 本质 / tool use 协议 / 错误处理是否吃透。30 秒内答得出 = 过。

---

## ReAct 与 Tool Use（对应任务 1）

1. **ReAct 和直接 chain-of-thought 的区别是什么？什么时候必须用 ReAct？**
   - 想点：CoT = model 自己 think → answer，纯文本；ReAct = think → act (tool call) → observe (tool result) → think → ...，循环；必须用 ReAct 的场景 = 答案依赖外部信息（文件 / API / database），model 内部没有

2. **Tool use loop 的 4 个核心步骤是什么？**
   - 想点：(1) 把 tools schema 传给 model；(2) model 返回 response、可能含 tool_use；(3) 如果 stop_reason="tool_use" → host 执行 tool → 把 result 当 user turn 加回 messages；(4) 回 (2)；如果 stop_reason="end_turn" → 返回最终答案

3. **为什么用框架（LangGraph / LangChain）反而对学习有害？**
   - 想点：框架把核心循环抽象掉、变成"配置 graph node"——表面上简洁，但隐藏了 tool use 的本质、调试时不知道发生了什么；ReAct 本质就是 200 行循环，自己写一遍才能讲清楚

4. **Single-turn function call 与 multi-step agent loop 的差别是什么？**
   - 想点：single-turn = 一次 API call、model 回一个 tool call、host 执行返回（如 OpenAI 早期 function calling）；multi-step = 循环、model 可以 tool call N 次直到 stop_reason="end_turn"；agent 必须 multi-step

---

## 错误处理（对应任务 1.5）

5. **Tool 失败时 agent 应该怎么处理？为什么不能在 host 侧 try/except 吞掉错误？**
   - 想点：把 exception stringify 后作为 tool_result 返回 → model 看到 "Error: ..." 自己决定重试 / 换 tool / 放弃 / 报告给用户；host 侧吞掉 → model 不知道失败了、可能继续基于错误前提推理

6. **Agent 怎么避免无限循环？**
   - 想点：(1) `max_iterations` 硬上限；(2) 检测 model 反复调同一 tool 同 args（probably 在死循环）；(3) cost cap（每个 task 限制总 token）；(4) 时间超时

7. **Tool 调用 args 经常错（比如 path 错了），怎么减少错误率？**
   - 想点：(1) tool description 写清晰、给 example；(2) input_schema 用 required + 严格 type；(3) 在 system prompt 里给行为指引（如"路径不确定时先 list_dir"）；(4) 错了让 model 自己看 result 改

---

## Context 管理（对应任务 1）

8. **Agent loop 跑久了 messages 会无限增长——怎么办？**
   - 想点：(1) 长输出 tool_result 截断（只保留前/后 N 行）；(2) 老 turn summarize 进 system prompt；(3) 关键决策用 structured output 落到外部（trace logger / state file）、context 里只留 reference；(4) 切到大 context window model

9. **System prompt vs first user message vs tool description——agent 行为塑造各靠哪个？**
   - 想点：system prompt = 角色 + 整体策略 + 不变的约束（"你是 X、必须 Y"）；first user message = 当前 task；tool description = 单个 tool 的语义和用法；调 prompt 时改对地方很关键

---

## Trace（对应任务 2）

10. **Trace logger 为什么要落 jsonl 而不是 print log？**
    - 想点：jsonl 结构化、方便后续 eval / 回放 / 调试；每行一个 record、可逐步 stream 写；C-M2 的 trajectory eval 直接吃这个格式

11. **Trace 里至少要记哪几类 step？**
    - 想点：(1) model_thought（每次 model 输出的 text）；(2) tool_call（name + args）；(3) tool_result（result + duration）；(4) 可选：cost (token 消耗)、final answer

---

## 自查标准

- 11 题里 ≥ 9 题 30 秒内答得出 → C-M0 过关
- 题 1-4（ReAct 本质）和 题 5-7（错误处理）必答好
