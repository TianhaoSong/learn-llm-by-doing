# C-M4 · Doing 任务详细规格

> Topic：工具设计的取舍——粒度、schema、安全边界、限速。**只写文档**，不实现新 agent。
>
> 一个任务：`slack_agent_tools.md` 工具表 + 安全设计文档。

---

## 任务 1 · `slack_agent_tools.md`

### 为什么做这个
agent 表现好不好，一大半取决于你给它的工具设计得好不好——工具粒度太粗 model 容易调错参数，太细又得来回调好多次；schema 描述含糊 model 就会瞎填；写操作没有 dry-run 和限速保护，agent 一旦抽风就可能真把消息发出去、把单子建出来。这个练习不写代码，是逼你像审 API 一样审一套真实工具：哪个该拆细、哪个 schema 要补 example、哪个写操作是高风险得加护栏。把这套取舍想透，你设计任何 agent 工具时才不会埋下隐患。

### 目标
把工作中实际 Slack agent 用到的工具列出来；每个工具评估粒度合理性、schema 设计、read/write 安全边界、限速处理；标注哪些工具是高风险。

### 必填结构

#### 1.1 工具表
对实际 Slack agent 的每个工具，列：
| Tool name | Description | Schema | R/W | Risk | Rate limit |
|---|---|---|---|---|---|
| search_messages | search Slack channels | (query, channel?, time_range?) | R | L | 50/min |
| get_thread | fetch full thread | (thread_ts, channel) | R | L | 100/min |
| post_message | post to channel | (channel, text) | W | M | 1/sec/channel |
| create_ticket | create JIRA from Slack | (project, title, body, assignee?) | W | **H** | 10/min |
| ... | | | | | |

#### 1.2 工具粒度审视
对每个工具，回答：
- 这个工具粒度对吗？粗了能省 round trip 但参数空间大、model 容易调错；细了调用次数多但每次决策简单
- **经验法则**：高频常用动作粗一些（search_messages 一个工具就能 cover 多种 query）；低频危险动作细一些（create_ticket 拆成 `validate_ticket_params` + `create_ticket_with_validation`）

#### 1.3 Schema 描述质量
对每个工具，检查：
- 每个参数有 `description` 吗？
- 关键参数（channel name vs id, time format）有 1-2 个 example 吗？
- enum 值用 `enum` 而不是用 description 列举吗？
- 必填字段标 `required` 吗？

#### 1.4 Read vs Write 边界
- 把 R 工具和 W 工具分开列，W 工具单独标注：
  - 是否有 dry-run mode？（write 工具必须支持 `dry_run: bool` 参数返回"会做什么但不真做"）
  - 是否有 confirmation step？（W 工具调用前最好让 agent 复述意图、用户/supervisor 确认）
  - 失败回滚？

#### 1.5 限速 / Quota
- 对每个 W 工具，标注 Slack API rate limit
- 工具内部应该做指数退避（hit 429 时 sleep + retry）
- **关键**：不要让 model 看到 429 错误（让它误以为该换工具）；host 侧拦截、retry transparently

### Code review checkpoint
- 至少有一个工具被标注 "粒度可能要拆细"——不要全 rationalize 现状
- 所有 W 工具都有 dry-run mode 设计 / 或明确说"目前没有，是 risk"
- 至少一个工具有 schema description 改进建议（"原始 description 不够 → 改成 X"）

### 失败排查
- **工具表不到 5 个**：Slack agent 还太初级；多列几个常见操作即使没实现（search_users、archive_channel）也算工具设计思考
- **Risk 列全 L**：危险——用户能创建 ticket / 发消息都是 M+；想清楚什么算 risk

### 辅助阅读（非 canonical）
- Slack API rate limits：https://api.slack.com/apis/rate-limits
- Anthropic tool description best practices：https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview

### Deliverable
- `course-c/m4-slack-tools/slack_agent_tools.md`（脱敏版可放公开 repo）
- 也可在项目内 `docs/tools.md` 放真实版

---

## 任务做完之后

- 跑 QUIZ.md（轻量）
- C-M4 过关 → 课程 C 完成 → learn-llm-by-doing 项目完整
