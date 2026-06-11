# C-M4 · 知识自查口试

> 验证工具设计取舍是否吃透。轻量模块、6 题足够。30 秒内答得出 = 过。

---

1. **粗工具 vs 细工具，怎么选？**
   - 想点：粗 = 省 round trip 但参数空间大、调错率高；细 = 调用次数多但每次决策简单；高频常用动作粗一些、低频危险动作细一些（强制 model 多步确认）

2. **Schema description 对 model 调用准确率影响多大？怎么写好？**
   - 想点：影响很大——参数加 description + 1-2 个 example 能减少 50%+ 参数错误；关键值用 `enum`、不要用 description 列举

3. **Read 工具 vs Write 工具的安全边界怎么设计？**
   - 想点：Write 工具必须有 (a) dry-run mode（参数 `dry_run: bool` 返回"会做什么但不真做"）；(b) confirmation step（调用前让 agent 复述意图）；(c) 最小权限 IAM；(d) 调用 trace 可审计；不要让 model 直接做不可逆 W 操作

4. **Slack API rate limit 怎么处理？为什么不能让 model 看到 429？**
   - 想点：工具内部做指数退避（hit 429 sleep + retry）；让 model 看到 429 它会以为该换工具或放弃；host 侧拦截 transparent retry、最多在 retry N 次后才把 "rate limited" 暴露给 model

5. **你 Slack agent 里哪个工具最容易被误用？怎么设防？**
   - 想点：通常是 `create_ticket` / `post_message` 这类 W 工具；设防：(a) dry-run + confirmation；(b) 最小权限（只能创建特定 project / 发到指定 channel）；(c) 限频；(d) audit log

6. **如果一个工具的调用错误率特别高（如 model 总传错 channel name），怎么办？**
   - 想点：(1) 拆细——分成 `list_channels` + `post_to_channel(channel_id)`、强制先查 id；(2) schema 加 `description: "Use channel ID (C0123...), not channel name"`；(3) 工具内部 fallback——传 name 时自动转 id；(4) 高频错的 case 加进 system prompt 例子

---

## 自查标准

- 6 题里 ≥ 5 题答得出 → C-M4 过关 → 课程 C 完成 → learn-llm-by-doing 全部模块完成
