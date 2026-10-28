# 3060 上 GRPO 的 kernel 级优化：结果

日期：2026-10-27。分支 `claude/gpu-kernels`，隔离环境 `~/.venvs/aha-kern`。计划见
`docs/superpowers/plans/2026-10-27-gpu-kernel-optimisation.md`，每项的原始数字在 `docs/kernels/`。

## 1. 结论

- **P1 Triton NF4 GEMM**：vLLM decode 每步 29.8 → 24.8 ms（1.20 倍）。GSM8K 全量配对评测 0.773 → 0.764，McNemar p = 0.24，差异不显著。
- **P3 fp16 累加 GEMM**：在 q/o 形状上达到 34.1 TFLOPS，超过 fp32 累加的 25.6 TFLOPS 上限，是 cuBLAS 的 1.54 倍，误差几乎不变。这是这次最有价值的发现。
- **P2 Liger 融合 loss**：在 Unsloth 下开关无效，Unsloth 本身已经分块计算 logprob。
- **P4 逐元素 kernel 归因**：之前看到的 fill/copy 大头是 profile 脚本自己造成的，训练真正执行的代码里这部分不到 3%，没有可优化的目标。

## 2. 基线（P0）

| 测什么 | 结果 |
|---|---|
| bnb 4-bit GEMM，decode 形状 M=6 | 12%–22% 带宽；每步合计约 20 ms，与 trace 一致 |
| 同形状 cuBLAS 读 bf16 权重 | 70%–80% 带宽，墙钟和 bnb 4-bit 一样 |
| cuBLAS bf16，训练形状 M=2400 | 大矩阵达到 fp32 累加峰值的 94%–102% |
| cuBLAS fp16 和 bf16 | 一样快：PyTorch 走的 cuBLAS 不用 fp16 累加 |

更正：之前把 3060 的 bf16 峰值写成 51 TFLOPS，实际 fp32 累加只有 25.6。所以"GEMM 只跑到 28% 峰值"不成立，cuBLAS 已经贴着上限。

## 3. Triton NF4 GEMM（P1）

- 设计：权重保持打包状态直到进寄存器；用 PTX `prmt` 字节置换树查 NF4 码表；split-K=4；先做 dot 再乘 absmax。
- 三轮设计空间探索（查表方式、tile、split-K、warps、stages）的结论是：反量化是 ALU 瓶颈，不是访存。
  去掉查表时 Triton 能到 40% 带宽，这就是这个设计在 Triton 里的天花板。
- 单 kernel：down 1.59 倍，q/o 1.21 倍，gate_up 1.06 倍；kv_proj 和 M ≥ 24 时比 bnb 慢，按 M ≤ 16、N ≥ 1024 路由。
- 端到端 decode 1.20 倍。GSM8K 评测墙钟只快 4.7%，因为评测里 prefill、LoRA 和提前结束的序列都不经过新 kernel。
- 默认关闭；`AHA_TRITON_NF4=1` 时 `scripts/train.py` 和 `scripts/eval.py` 启用。

## 4. Liger 融合 loss（P2）

Unsloth 的 `compute_loss` 永远走 `grpo_accumulated_loss`，从不调用 TRL 的 Liger 分支。开和关的步时、峰值显存（7.67 GB）都相同。
自写 lm_head + loss 融合的收益也小，因为 Unsloth 已经不落地完整的 fp32 logits。

## 5. 逐元素 kernel 归因（P4）

最大的两行（copy 648 ms、fill 412 ms，各 4 次 launch）正好是 [6×400×151936] fp32 logits 的 cast 和梯度清零，
来自 `profile_step.py` 里的朴素 `log_softmax`，真实训练不会产生。能归到训练代码的部分合计不到 kernel 时间的 3%。

## 6. fp16 累加 GEMM（P3）

| 形状 | cuBLAS | Triton fp16 累加 | 倍数 | 相对误差 |
|---|---|---|---|---|
| q/o 2048×2048 | 22.2 TFLOPS | 34.1 | 1.54× | 0.0005（cuBLAS 0.0004） |
| gate_up | 24.7 | 32.0 | 1.30× | 0.0005 |
| down | 26.0 | 25.9 | 1.00× | 0.0005 |

每个 32 宽的 K 块就提升到 fp32 一次，速度没有损失，误差基本不变。结论：有条件 go。

## 7. 下一步（按收益排序）

1. **P3 落地**：按形状 autotune（down 用 split-K）；把前向的四个底座 GEMM 放到开关后面，模型改 fp16 + loss scaling，用 smoke 配置对比 10 步的 loss 和 reward。
   训练半步 70% 是 GEMM，如果平均 1.3 倍，训练半步约快 15%。
2. **P1 噪声底**：再跑一次 bnb 对 bnb 的 GSM8K 全量评测（33 min），确认 −0.9 个点落在采样噪声里。
3. **P1 CUDA 版**：共享内存 LUT + `cp.async` + `ldmatrix`，突破 Triton 的 40% 天花板；同时把 LoRA 的 x·A·B 融进 epilogue。
4. 用 `AHA_TRITON_NF4=1` 跑一次完整的 500 步训练，对比步时和精度。
