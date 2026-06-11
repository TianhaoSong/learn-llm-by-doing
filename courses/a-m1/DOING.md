# A-M1 · Doing 任务详细规格

> Topic：把 transformer 从"读过 paper"变成"能徒手写出来"。GPT-2 small（~124M）单卡 + TinyShakespeare 跑通 + 与参考实现数值对照。
>
> 三个任务对应三个层面：`mygpt/` = 自己写一遍结构 + 训练循环；`test_attention.py` = 数值对照证明实现正确；`torch.compile` 实验 = 体感 graph capture 的收益。

---

## 任务 1 · `mygpt/` 单卡训练 GPT-2 small

### 为什么做这个
读 paper 时 attention、residual、weight tying 这些都"看懂了"，但真要你不查资料把整个 GPT 从空白文件敲出来，多半会卡在 shape 怎么 reshape、mask 加在 softmax 前还是后这种细节上。这些细节恰恰是后面所有事情的地基——你要在它上面加 DDP、加混精度、加 FSDP，如果连模型本身都不是自己写的，出了 bug 你永远分不清是分布式的锅还是模型的锅。亲手在 TinyShakespeare 上跑到能续写出像样的文本，才算真的"会"transformer，而不是"见过"。

### 目标
不查 nanoGPT 能徒手写出 multi-head causal self-attention、LayerNorm、residual、FFN、token + position embedding 的完整 GPT-2 结构。在 TinyShakespeare 上跑通到 loss 收敛、能续写连贯文本。

### Sub-tasks（推荐顺序）
1. **Tokenizer**：`import tiktoken; enc = tiktoken.get_encoding("gpt2")`——直接复用 OpenAI BPE，不自己训。包一个 `mygpt/data.py`：把 `input.txt` encode 成一维 token 数组，splits 成 train (90%) / val (10%) 落 `train.bin` / `val.bin`（用 `np.memmap` 之后训练时读）
2. **Config**：写 `mygpt/config.py`——`@dataclass GPTConfig(vocab_size=50257, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0)`（先用比 GPT-2 small 还小的版本跑通；最后一步再切到 124M 配置 `n_layer=12, n_head=12, n_embd=768, block_size=1024`）
3. **Model 各层**（一个一个写，每写一个手算一次 shape）：
   - `CausalSelfAttention`：QKV linear → split heads `(B, T, n_embd) → (B, n_head, T, head_dim)` → 算 `att = q @ k.T / sqrt(head_dim)` → 加 causal mask（上三角填 `-inf`）→ softmax → `att @ v` → merge heads → output linear
   - `MLP`：`Linear(n_embd, 4*n_embd)` → `GELU` → `Linear(4*n_embd, n_embd)` → dropout
   - `Block`：`x = x + attn(LayerNorm(x))`、`x = x + mlp(LayerNorm(x))`（pre-norm，跟 GPT-2 一致）
   - `GPT`：`wte` (token embedding) + `wpe` (position embedding) → 多个 Block → final LayerNorm → `lm_head`（注意：`lm_head.weight = wte.weight` 是 weight tying，省 ~30M 参数）
4. **Forward 返回 (logits, loss)**：训练时传 `targets`，推理时只传 `idx`
5. **Generate 函数**（推理用，先简单 greedy / temperature sampling，KV cache 留到 B 课程）：循环 N 步，每步 `forward → logits[:, -1, :] / T → softmax → multinomial sample`；每步把 idx 截断到最后 `block_size` token
6. **Train loop**：
   - 数据：`np.memmap('train.bin', dtype=np.uint16)` + 每 step 随机抽 `batch_size` 个起点拿 `block_size` 长度
   - Optimizer：`AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)`；可选 cosine decay
   - 跑 ≥ 5000 step，每 200 step 算一次 val loss + 抽样 generate 看一眼
7. **小配置先跑通**：`n_layer=6, n_head=6, n_embd=384, block_size=256`，单卡几分钟内 train loss 应明显下降、val loss 先降后回升（过拟合）。跑通后再切 124M 配置看显存和 throughput

> **关于 loss 数字（重要）**：本任务用 **tiktoken BPE（vocab=50257）**，不是 char-level（vocab≈65）。两者 loss 不可比——网上常见的 "TinyShakespeare loss ~1.5" 是 **char-level** 的数字，BPE 上根本到不了。
> BPE + TinyShakespeare（只有 ~270K train token）的真实预期：**val loss 触底 ~4.7–5.0 就过拟合反弹**（数据太少、模型记住了训练集），这是数据量天花板、不是 bug。想压更低只能换 char-level tokenizer 或更多数据（A-M4 的 FineWeb）——都不是本任务该做的。
> **过关看的是趋势 + 生成质量，不是绝对 loss 数字。**

### 成功标准
- TinyShakespeare（BPE）上 train loss 单调下降、val loss 先降后回升（过拟合是预期）；**做到 val ~5 就算正常**，不追绝对数字，肉眼看下降趋势即可
- 抽样能生成**语法基本通顺**的莎士比亚风格文本（句子有标点、有大小写、不是纯乱码）
- Code review checkpoint：
  - causal mask 用 `register_buffer` 注册（不是每次 forward 重建）
  - softmax 之前 `att = att.masked_fill(mask == 0, float('-inf'))`（不是 `att * mask`）
  - `lm_head.weight = wte.weight` 做了 weight tying
  - `forward` 接 `targets=None`：传 None 时返回 `(logits, None)`、传 targets 时返回 `(logits, loss)`

### 失败排查
- **Loss 不动卡在 ~10**：vocab_size 错了（必须 50257），或者 logits 没和 targets 在同一 vocab 空间
- **Loss 第一步就 NaN**：attention scores 没除 `sqrt(head_dim)`，softmax 数值溢出
- **Generate 输出全是同一个字符**：sampling 没做 temperature / top-k；或者 greedy 在小模型 + 短训练下确实会 mode-collapse，调 `temperature=0.8 + top_k=200`
- **Loss 降到 1.0 以下但 generate 是乱码**：忘了切 `model.eval()`，dropout 还开着；或者 generate 时 idx 没截断到 block_size，导致 position embedding 越界
- **显存爆**：先调小 `block_size`（256→128）和 `batch_size`，确认其他都对再放大

### 辅助阅读（非 canonical）
- nanoGPT `model.py`（卡住时对照看，**不要直接抄**）：https://github.com/karpathy/nanoGPT/blob/master/model.py
- nanoGPT `prepare.py`（TinyShakespeare 数据处理模板）：https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare/prepare.py
- tiktoken README（`encoding_for_model("gpt-2")` 用法）：https://github.com/openai/tiktoken

### Deliverable
- `course-a/m1-mygpt/mygpt/{config.py, model.py, data.py, train.py}`
- `course-a/m1-mygpt/sample_output.txt`：最终 checkpoint 上 generate 一段 ~500 token，贴在文件里
- `course-a/m1-mygpt/notes.md`：踩坑 + 数值/形状容易搞错的地方记一两句

---

## 任务 2 · `tests/test_attention.py` 数值对照

### 为什么做这个
你自己写的 attention "loss 在降"不代表它真的对——很多实现 bug（scale 漏了 sqrt、mask 写反）会让 loss 照样下降，只是模型偷偷学歪了。等到后面跑多卡分布式出问题，你最不想做的就是怀疑"是不是我模型本身就错了"。所以现在趁实现还简单，用 PyTorch 官方的 fused attention 当标准答案做一次逐元素对照，把"我的 attention 数值上就是对的"这件事钉死，后面调试时就能安心排除这一项。

### 目标
用 `F.scaled_dot_product_attention`（PyTorch 内置 fused kernel，可信 ground truth）作对照，证明你写的 attention 实现数值正确。这是一个"信心测试"——后续 A-M2 跑分布式调试时不会怀疑模型本身有 bug。

### Sub-tasks
1. 构造一个固定 seed 的输入：`B=2, T=16, n_embd=64, n_head=4, head_dim=16`，fp32
2. 把你的 `CausalSelfAttention` 实例化，extract 它的 QKV linear 权重
3. 手动算一遍 Q/K/V → split heads → 调 `F.scaled_dot_product_attention(q, k, v, is_causal=True)` 得到 `ref_out`
4. 调你自己的 attention forward 拿 `my_out`
5. assert：`torch.allclose(my_out, ref_out, atol=1e-5, rtol=1e-5)` 在 fp32 下成立
6. 加一组 dropout=0 的边界 case；dropout > 0 不要测（随机性对不上）

### 成功标准
- `pytest tests/test_attention.py` 通过
- 数值差 ≤ 1e-5（fp32）；如果是 1e-3 量级要排查（很可能是 mask 写错或者 scale 漏了 sqrt）

### 失败排查
- **数值差 ~0.1**：scale 漏了 / softmax 之前没减最大值（PyTorch 内置 SDPA 数值稳定性更好，自己实现也要减最大值）
- **数值差 ~1e-3**：output projection 的权重你忘了 apply（test 里只比较 attention 主体输出，不要包含 output linear）
- **shape 不一致**：split heads 的 reshape 顺序错了——QKV 投影后的 reshape 应该是 `(B, T, n_head, head_dim).transpose(1, 2)` 得到 `(B, n_head, T, head_dim)`

### 辅助阅读
- `F.scaled_dot_product_attention` 文档：https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

### Deliverable
- `course-a/m1-mygpt/tests/test_attention.py`

---

## 任务 3 · `torch.compile` step time 实验

### 为什么做这个
`torch.compile` 经常被当成"加一行就免费提速"的魔法，但实际上它有明显的冷启动代价（第一次要编译几十秒），而且收益高度依赖你的模型规模和是否已经开了 TF32／混精度——小模型上可能只快几个百分点。亲手测一组前后对比的数字，你才会对"什么时候该用、能指望多少收益"有真实体感，而不是迷信或盲目套用。这个工具后面几个模块都会反复用到，现在把它的脾气摸清楚很值。

### 目标
体感 `torch.compile` 把 graph capture + kernel fusion 带来的收益（也看到它的代价：第一次跑会编译 30s+）。这是后续 A-M2 / A-M3 / B 课程都会用到的工具。

### 值得理解（compile 不是免费午餐）
- **cold start 开销很明显**：A100 上图编译 + Inductor codegen + kernel autotune 一次性 ~50s（25M 模型）。这就是为什么要丢 warmup、只看 warm median。
- **warm 收益可能很小（甚至 10% 都不到），三个原因**：
  1. **模型太小**：`torch.compile` 主要省 Python 调度 + kernel launch overhead。小模型（25M）单 op 的 GPU kernel 本身就快，这块 overhead 占比低，fusion 收益天花板低。收益在**大模型 / 多 kernel / 小 batch**（launch overhead 占比大）时才显著。
  2. **没开 TF32**：A100/H100 有 TF32 tensor core，但 PyTorch 默认 fp32 matmul **不启用**它（走慢路径）。这时即使 compile fuse 了 kernel，底层 matmul 还是慢的，相对收益被压死。**`torch.set_float32_matmul_precision('high')` 单独就能给 no-compile 也大幅提速**——先开 TF32 再比 compile，结论才公平。
  3. **没用混精度**：fp32 下 memory bandwidth 和 tensor core 都没吃满（bf16 autocast 是 A-M2 的内容）。
- **结论**：compile 的收益取决于「模型规模 + 你是否已经在用 TF32/混精度」。在已经 TF32+bf16 的大模型上，compile 才是明确的赢；在 fp32 小模型上可能只快个位数 %。

### Sub-tasks
1. 在 `train.py` 里加一个 `--compile` flag
2. 不开 compile 跑 100 step（前 5 步 warmup 丢掉），用 `torch.cuda.Event` 计 median step time
3. 开 compile（`model = torch.compile(model)`）跑 100 step：**前 5 步包含编译耗时单独打出来**，后 95 步取 median
4. 对比表格写进 README：`(no compile, compile-cold, compile-warm) ` 三列
5. 重要观察：第一步 compile 的耗时（通常 20-60s，看模型大小）

### 成功标准
- 给出三个数字：no-compile median / compile 第 1 步耗时 / compile warm median
- compile-warm 应当 ≤ no-compile（如果不快，可能是模型太小、kernel launch overhead 不是瓶颈——这种情况说明也写进 README）
- README 一段话解释"为什么 compile 有收益"——提到 kernel fusion、减少 Python overhead

### 失败排查
- **compile-warm 只快一点点 / 反而慢**：见上方「值得理解」——多半是 (a) 模型太小、(b) 没开 TF32。先加 `torch.set_float32_matmul_precision('high')` 重测；模型小是固有限制，记进 README 即可，不用强行优化
- **看到 `TensorFloat32 ... not enabled` warning**：A100/H100 上没启用 TF32 tensor core，matmul 走 fp32 慢路径。加 `torch.set_float32_matmul_precision('high')`（或 `'medium'`）。这一行常常比 compile 本身的提速还大
- **compile 第一步报错**：常见 `dynamic shape` 问题，每个 batch 长度不一致——TinyShakespeare 训练时 block_size 是固定的，应该没这问题；如果有，加 `torch.compile(model, dynamic=False)`
- **compile 第一步耗时 > 5 分钟**：可能 fallback 到了不必要的重编译——检查是不是每次 forward 输入 shape 都在变

### 辅助阅读（非 canonical）
- PyTorch `torch.compile` 教程：https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

### Deliverable
- `course-a/m1-mygpt/compile_results.md`：
  - 表格：(no-compile, compile-cold, compile-warm) 的 step time / tok/s
  - 一段解释「为什么 compile 有收益」（kernel fusion + 减少 Python overhead）
  - 一段归因「warm 收益为什么是这个量级」（模型规模 / TF32 / 混精度——见上方「值得理解」）
  - 加分项：开 `torch.set_float32_matmul_precision('high')` 后重测一组，看 TF32 单独的提速 vs compile 的提速谁大

---

## 三个任务做完之后

- 自答 QUIZ.md，14 题里至少 12 题能在 30 秒内答出来
- 答不出的回去查任务 / 学习材料
- A-M1 过关 → 开 A-M2（多卡 DDP）
