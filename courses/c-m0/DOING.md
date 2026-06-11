# C-M0 · Doing 任务详细规格

> Topic：≤ 200 行 raw API 循环实现 ReAct / tool use 本质。不用 LangChain/LangGraph 框架——直接写 model → tool call → result → model 的循环。
>
> 两个任务：`mini_agent.py` + trace logger。

---

## 任务 1 · `mini_agent.py` — Raw API tool use 循环

### 为什么做这个
所有 agent 框架（LangChain、LangGraph、autogen）底下都是同一个循环：把任务给 model → model 说"我要调某个工具" → 你执行工具、把结果喂回去 → model 再决定下一步，直到它说"做完了"。框架把这个循环包了一层糖，你看不清里面发生了什么。自己用 raw API 手写一遍，你才会真正理解 ReAct 到底是什么、tool call 是 model 输出的一段结构化数据而不是什么魔法，以后调框架出问题时你知道往哪看。

### 目标
写一个 ≤ 200 行的 mini agent：单 agent + 3 个工具（read_file / list_dir / run_shell），跑"找出仓库里所有 TODO"任务。**不用框架**——直接调 Anthropic / OpenAI API。

### Sub-tasks

#### 1.1 环境
```bash
pip install anthropic python-dotenv  # 或 pip install openai
```
- API key 放 `.env`：`ANTHROPIC_API_KEY=...` 或 `OPENAI_API_KEY=...`
- 用 `dotenv.load_dotenv()` 读
- **绝不**写进代码 / commit；这是合规要求

#### 1.2 定义 3 个工具（host 侧 Python 函数）
```python
TOOLS = [
    {
        "name": "read_file",
        "description": "Read the content of a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "run_shell",
        "description": "Run a shell command. Use cautiously.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

def execute_tool(name, args):
    try:
        if name == "read_file":
            return open(args["path"]).read()
        elif name == "list_dir":
            return "\n".join(os.listdir(args["path"]))
        elif name == "run_shell":
            return subprocess.check_output(args["command"], shell=True, text=True)
    except Exception as e:
        return f"Error: {e}"  # 失败也返回字符串、让 model 自己处理；不要 raise
```

#### 1.3 Agent 主循环
```python
import anthropic
client = anthropic.Anthropic()

def run_agent(task):
    messages = [{"role": "user", "content": task}]
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",  # 或当时最新版
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )
        # response.content 可能含 text + tool_use blocks
        messages.append({"role": "assistant", "content": response.content})
        
        if response.stop_reason == "end_turn":
            return final_text(response)
        
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
            messages.append({"role": "user", "content": tool_results})
            continue
        
        # 其他 stop_reason: max_tokens 等——根据情况处理
        break
```

#### 1.4 跑 demo 任务
- 调用：`run_agent("Find all TODO comments in the current directory and list which file they're in.")`
- 预期 agent 会：list_dir → 看到 .py 文件 → read_file 几个 → run_shell `grep -r TODO .` → 返回结果

#### 1.5 控制循环上限
- 加 `max_iterations=20`，防止无限循环烧钱
- 加 cost 估算：每次 API call 后打 `response.usage.input_tokens + output_tokens`，方便看烧了多少 token

### 成功标准
- ≤ 200 行代码（含注释）跑通"找 TODO"任务
- 输出是 agent 调了几次 tool 后给出的 final answer，不是单次 API response
- 不依赖 LangChain / LangGraph / autogen 等框架——只 import anthropic（或 openai）+ stdlib

### 失败排查
- **agent 一直循环不停**：max_iterations 没设；或者 model 一直返回 `tool_use` stop_reason 不收敛——通常是 task 太开放、tool description 不清楚；改 task 更具体
- **tool 调用 args 错误**：`input_schema` 没设 required 或者 description 写得不清楚；改 description（如 `"path": {"type": "string", "description": "Absolute or relative path; pass current dir as '.'"}`）
- **tool 失败后 agent 卡住**：execute_tool 不要 raise——必须返回字符串错误；让 model 看到 "Error: ..." 自己决定重试 / 换 tool / 放弃

### 辅助阅读（非 canonical）
- Anthropic tool use 的端到端示例：https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use
- OpenAI function calling（如用 OpenAI）：https://platform.openai.com/docs/guides/function-calling

### Deliverable
- `course-c/m0-mini-agent/mini_agent.py`
- 运行截图或输出贴在 `demo_output.md`

---

## 任务 2 · Trace logger

### 为什么做这个
agent 跟普通函数不一样：它每次跑的路径都可能不同，出问题时你光看最终输出根本不知道哪一步走歪了。把每一步（model 想了什么、调了哪个工具、工具返回了什么）都落成结构化日志，你才有办法事后复盘"它为什么绕了三圈"。而且这套 trace 格式后面做 eval 时要直接拿来用——你没法评估一个看不见过程的 agent。

### 目标
在工具调用上加一层 trace logger，每一步落 jsonl。这套 log 格式 C-M2 会复用做 trajectory eval。

### Sub-tasks
1. 加 `trace.py`：
   ```python
   import json, time
   def log_step(trace_file, step_idx, step_type, payload):
       record = {
           "step": step_idx,
           "type": step_type,  # "model_thought" | "tool_call" | "tool_result"
           "ts": time.time(),  # 注意：实际项目不用 time.time()，用 datetime.utcnow().isoformat()
           **payload,
       }
       with open(trace_file, "a") as f:
           f.write(json.dumps(record) + "\n")
   ```
2. 在 `run_agent` 里每个关键节点调 `log_step`：
   - 每次 model response 落 `model_thought`（response 里所有 text block 拼起来）
   - 每次 tool 调用前落 `tool_call`（name + args）
   - 每次 tool 返回后落 `tool_result`（result + duration）
3. 一次 agent 跑完一个 `trace_<task_id>.jsonl`

### 成功标准
- 跑完 demo 任务后产出一个 jsonl，每行一个 step
- 类型 (`model_thought` / `tool_call` / `tool_result`) 都至少出现一次
- 时间戳 + duration 字段都有，方便后续算 latency

### Deliverable
- `course-c/m0-mini-agent/trace.py`
- 一个示例 `trace_demo.jsonl`

---

## 两个任务做完之后

- 跑 QUIZ.md
- C-M0 过关 → 开 C-M1（Multi-agent 架构决策）
