# A-M1 · 知识自查口试

> 验证 transformer 内部机制 / KV cache 直觉 / 数值稳定性是否吃透。30 秒内答得出 = 过。

---

## Attention 机制（对应任务 1 + 任务 2）

1. **Causal mask 是怎么实现的？为什么是上三角填 `-inf` 而不是 `0`？**
   - 想点：上三角对应 "未来位置"；softmax(-inf) = 0 严格屏蔽；填 0 会让概率分布偏向未来 token

2. **Softmax 之前为什么减最大值？数值稳定性具体是什么？**
   - 想点：`exp(x)` 在 x=88 左右 fp32 就溢出 inf；减 max 后 exp 输入 ≤ 0、范围 (0, 1]，结果不变（softmax 对加常数不变）

3. **Attention 的 scale 因子 `1/sqrt(head_dim)` 是为了什么？不除会怎样？**
   - 想点：q·k 是 head_dim 个独立项之和，方差 ~head_dim、std ~sqrt(head_dim)；不除会让 softmax 输入数量级太大、变成 one-hot、梯度消失

4. **Multi-head attention 的 `head_dim = n_embd / n_head` 这个除法是必须的吗？为什么不让每个 head 都用全 n_embd？**
   - 想点：是惯例不是必须；split 是为了让总 FLOPs 和参数量与 single-head 相当（`n_head × head_dim² ≈ n_embd²`）；让 head 都用全 n_embd 会贵 n_head 倍

---

## Transformer 结构（对应任务 1）

5. **Pre-norm vs post-norm 的区别？GPT-2 用哪个，为什么？**
   - 想点：pre-norm 是 `x + attn(LN(x))`，post-norm 是 `LN(x + attn(x))`；pre-norm 训练更稳定（梯度直接走 residual 不经 LN），深层模型几乎都用 pre-norm；GPT-2 用 pre-norm

6. **Weight tying（`lm_head.weight = wte.weight`）是什么？为什么 GPT-2 这么做？**
   - 想点：输入 embedding 矩阵和输出投影共享权重；省 ~30M 参数（`vocab_size × n_embd`）；隐含假设"语义相似的 token 在 embedding 空间相似"

7. **Position embedding 为什么需要？GPT-2 用哪种？RoPE / ALiBi 跟它差别在哪？**
   - 想点：attention 本身对 token 顺序无感（permutation invariant），需要外加位置信息；GPT-2 用**learned absolute** position embedding（`wpe`）；RoPE 把位置信息编进 Q/K 的旋转里、外推性好；ALiBi 给 attention scores 加距离 bias、不引入参数

8. **`block_size` 在训练时是什么？推理时呢？为什么 generate 要截断到最后 `block_size` token？**
   - 想点：训练时是 context length；推理时也是 max context（position embedding 的最大下标）；超过会越界（`wpe[idx]` 取不到）

---

## KV Cache（A-M1 埋的伏笔，B 课程展开）

9. **KV cache 在训练时存在吗？为什么？**
   - 想点：不存在；训练每个 step 输入是 `(B, T)` 完整序列，attention 一次性算所有位置；KV 算完就丢

10. **推理时 KV cache 的形状是什么？为什么需要它？**
    - 想点：`(num_layers, 2, batch, n_head, max_seq_len, head_dim)`，2 是 K/V；不缓存的话每生成一个 token 要把前面所有 token 重算一遍，O(N²) 变 O(N)（每步只算新 token 的 Q 和已缓存的 K/V）

11. **Decode 阶段每步只算一个 token 的 Q——这意味着 attention 的 shape 变成了什么？**
    - 想点：Q 是 `(B, n_head, 1, head_dim)`，K/V 是 `(B, n_head, T_so_far, head_dim)`；attention scores 是 `(B, n_head, 1, T_so_far)`；softmax 只对 T_so_far 做（不需要 causal mask，因为本来就只看过去）

---

## 训练循环 & 数据（对应任务 1）

12. **TinyShakespeare 训练时为什么用 `np.memmap` 而不是把文件 load 进内存？**
    - 想点：mmap 让 OS 按需 page in，启动快、不占 Python 进程内存；当 dataset 大到 ≥ 几 GB 时这是必须的（A-M4 的 FineWeb tokenized 是 GB 级）

13. **AdamW 的 `weight_decay=0.1` 应不应该作用到 LayerNorm 和 bias？**
    - 想点：不应该；LN 的 gain/bias 和所有 bias 都不该 decay（decay 它们会破坏归一化）；nanoGPT 标准做法是 split 成两组 param group——只对 dim ≥ 2 的张量 decay

14. **`torch.compile` 把"编译开销"换来了什么？什么场景不值得开？**
    - 想点：换 kernel fusion + 减少 Python 调度 overhead；模型小 / 单步太快（<1ms）/ 输入 shape 老变（dynamic）时 overhead 比收益大，不值得开

---

## 自查标准

- 14 题里 ≥ 12 题 30 秒内答得出 → A-M1 过关
- 第 9-11 题（KV cache）不会很正常——那是 B 课程的核心，A-M1 只要求知道"训练时不存、推理时存什么"的轮廓
