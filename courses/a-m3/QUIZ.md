# A-M3 · 知识自查口试

> 验证显存账本 / ZeRO Stage / FSDP 通信 / activation checkpointing 是否吃透。30 秒内答得出 = 过。

---

## 显存账本（对应任务 1）

1. **训练 1B 参数模型 + AdamW + bf16 mixed precision，光是 model state（params + grads + opt-state）大概多少 GB？**
   - 想点：`6 + 4 + 8 = 18` 字节 / 参数；1B × 18 = 18 GB；A100 40GB 单卡都装不下 model state，必须 ZeRO/FSDP

2. **AdamW 的 optimizer state 为什么是 `8 × n_params` bytes？SGD with momentum 是多少？**
   - 想点：AdamW 每参数有 fp32 的 first moment m + second moment v = 8 字节；SGD with momentum 只有 m = 4 字节；plain SGD = 0

3. **mixed precision 下 params 为什么是 `6 × n_params` 而不是 `2 × n_params`？**
   - 想点：bf16 副本（forward/backward）2 字节 + fp32 master 副本（optimizer step）4 字节 = 6 字节；optimizer 必须用 fp32 副本才数值稳定

4. **Activation 显存为什么很难精确算？为什么一般占整个训练显存的一大半？**
   - 想点：每层都要保存 forward 中间结果给 backward 用（attention scores、layernorm input、dropout mask 等）；公式有 ~10 项加在一起、constant 看实现；和 batch×seq×n_layer×n_embd 成正比，模型一大一深就远超 model state

5. **`torch.cuda.memory_summary()` 看到的是当前快照，要测 peak memory 应该用什么？**
   - 想点：`torch.cuda.reset_peak_memory_stats()` 清零，跑一段，再读 `torch.cuda.max_memory_allocated()`

---

## ZeRO Stage（对应任务 2-3）

6. **ZeRO Stage 1 / 2 / 3 各自切了哪三块的哪一块？**
   - 想点：Stage 1 切 optimizer state；Stage 2 切 optimizer state + grads；Stage 3 切 optimizer state + grads + params；activation 不在 ZeRO 范围（要靠 checkpointing 或 sequence parallel）

7. **为什么 ZeRO-1 几乎免费（通信不增加）但收益小？**
   - 想点：optimizer state 只在 step 那一刻用，原本就在每个 rank 上独立计算；Stage 1 把它切了之后 step 时只更新自己那部分 params，再 all-gather params 同步给所有 rank——相当于 step 后多一次 all-gather；DDP 本来就有 step——通信量没增加多少；收益是省 8/18 ≈ 44% 的 model state

8. **ZeRO-3 / FSDP FULL_SHARD 的 forward 通信模式是 all-gather，backward 是什么？为什么？**
   - 想点：forward 时每层进入前 all-gather params 算完释放；backward 时每层进入前再 all-gather params + 算完后 reduce-scatter grads（不是 all-reduce，因为 grad 也是切的，每个 rank 只要自己那段）；通信总量是 DDP 的 1.5 倍（多一次 all-gather）

9. **为什么 ZeRO-3 能训得动 DDP 训不动的模型？代价是什么？**
   - 想点：DDP 每张卡都要装下完整 params + grads + opt_state（18N 字节）；ZeRO-3 切了之后每张只要 18N/G + 临时 all-gather buffer（一层 params）；代价是通信量 1.5 倍 + 实现复杂度高

---

## FSDP 实现（对应任务 2）

10. **FSDP 的 `auto_wrap_policy` 不设会怎样？为什么必须按 transformer block wrap？**
    - 想点：默认整个模型当一个 shard 单元；forward 进来要把所有 params all-gather 到一起、算完再释放——和 DDP 一样占显存，毫无收益；按 block wrap 后每次只 all-gather 一层、算完释放、再 gather 下一层

11. **FSDP1 和 FSDP2 API 的本质差别是什么？**
    - 想点：FSDP1 是 monolithic wrapper（`FSDP(model, ...)` 包整个模型）；FSDP2 是 per-module（每个子模块自己 `fully_shard()`）；FSDP2 和 tensor parallel 组合更顺、控制更细；新项目推荐 FSDP2

12. **FSDP `MixedPrecision` 配置里的 `param_dtype` / `reduce_dtype` / `buffer_dtype` 各是什么？**
    - 想点：`param_dtype` = forward/backward 用的 dtype（bf16）；`reduce_dtype` = grad reduce-scatter 用的 dtype（bf16 也行，节省通信）；`buffer_dtype` = 模型 buffer（如 BN running stats）的 dtype

13. **FSDP 训练下 `SHARDED_STATE_DICT` 和 `FULL_STATE_DICT` 各是什么？什么时候用哪个？**
    - 想点：sharded = 每 rank 只存自己那部分 params + opt_state，存盘快、resume 快、必须用同样 N_GPU 才能 load；full = rank 0 把所有 rank 的 shard gather 起来存成完整文件，慢但跨 N_GPU 兼容；训练 checkpoint 用 sharded，最终 export 给推理用 full

---

## Activation Checkpointing（对应任务 2）

14. **Activation checkpointing 是怎么省显存的？代价是什么？**
    - 想点：forward 时不存中间 activation，只存每个 checkpoint 段的输入；backward 进入该段时重新跑一遍 forward 算 activation；代价是 forward 算了两次（理论 33% 多 compute，实际 throughput 降 ~20-30%）

15. **`use_reentrant=True` vs `False` 的差别？为什么 FSDP 推荐 False？**
    - 想点：reentrant=True 是老实现，用 autograd 的 backward hook，与某些功能（如 FSDP）配合有 bug；reentrant=False 是新实现，用 saved tensors API，更稳

16. **什么场景不该用 activation checkpointing？**
    - 想点：(1) compute-bound 训练（GPU 算力是瓶颈不是显存）；(2) 模型小（活动本来就不多）；(3) 显存还有余量；checkpointing 是显存救火不是常态

---

## 综合（跨任务）

17. **350M 模型在 8×A100 40GB 上从 OOM 到能跑 1024 batch，你应该按什么顺序加优化？**
    - 想点：(1) 先 bf16 mixed precision（先 free 一半 model state）；(2) 然后 FSDP SHARD_GRAD_OP；(3) 还不够上 FULL_SHARD；(4) activation 太大上 checkpointing；(5) 再不够 sequence parallel / 减 batch；先做收益大、代价小的

18. **如果你的 1B 训练 throughput 只有理论 30%（看 MFU），最可能的瓶颈是什么？怎么验证？**
    - 想点：通信—— FSDP 的 all-gather 没和 forward overlap；用 profiler 看 timeline、看 NCCL ops 占比；30% MFU 在 8×A100 上算合理基线，再低就是有问题

---

## 自查标准

- 18 题里 ≥ 15 题 30 秒内答得出 → A-M3 过关
- ZeRO Stage 通信差异（题 6-9）和 FSDP 实现细节（题 10-13）是面试核心
