# GPU kernel 级优化（3060 上的 GRPO）设计

日期：2026-10-27。硬件：RTX 3060 12 GB（GA106，WSL2）。模型：Qwen2.5-3B-Instruct，bnb NF4 + LoRA r=32，vLLM 0.10.2 同进程采样。

## 目标

把一步 GRPO 的两半（训练前向反向、vLLM decode）里占大头的 kernel 换成更接近硬件上限的实现，
每一项收益都用独立的 microbenchmark 量化，而不是只看端到端步时。方向限定为 kernel / 精度 / 显存流量，
不改训练配方（采样数、KL、评测 batch 这些另有计划）。

## 现状（来自 `results/profile/`，2026-10-26 的 trace）

硬件上限（GA106）：显存带宽 360 GB/s；tensor core fp16 累加 51 TFLOPS，**fp32 累加 25.6 TFLOPS**（GeForce 减半）；CUDA core fp32 12.7 TFLOPS。

训练半步（6 条 × 600 token，5.3 s kernel，9,360 次 launch，kernel 区间内 GPU 空闲仅 4%）：

| 类别 | 时间 | 说明 |
|---|---|---|
| 底座 bf16 GEMM | 2.9 s | 14.4 TFLOPS = fp32 累加峰值的 56% |
| fill / copy / cast 逐元素 | 0.75 s | 3,360 次 launch，纯显存流量，来源未归因 |
| DtoD memcpy + 词表 softmax 反向 | 0.49 s | [2400 × 151936] fp32 logits 的落地与回传 |
| LoRA 矩阵乘 | 0.04 s | fp32 sgemm，在 CUDA core 上 |

采样半步（每个 decode 步约 30 ms，CUDA graph 已开，墙钟 ≈ kernel）：

| 类别 | 每步 | 说明 |
|---|---|---|
| bnb `gemm_4bit_sm80` | 22 ms，252 次 | 读 2.35 GB 权重，107 GB/s，带宽利用 30% |
| punica LoRA、逐元素、zeros | 2.4 ms | glue |
| attention + KV 写入 | 0.8 ms | 不是瓶颈 |

## 四个工作项

**P0 microbenchmark 框架**（其它项的前提）：每个 kernel 单独计时（CUDA event，中位数，轮换多份权重打掉 L2），
输出达到的 GB/s、TFLOPS、相对硬件上限的百分比；对照 bnb / cuBLAS；和 fp32 参考比误差，阈值固定写进测试。
启动前检查 GPU 是否被别的进程占用。报告写到 `docs/kernels/`（`results/` 在 .gitignore 里）。

**P1 小 M 的 NF4 weight-only GEMM（Triton）**：decode 时 M = 6～24。设计：按 N 分块，K 按 64（= NF4 blocksize）分块，
packed uint8 直接加载，高低 nibble 拆成两组用两次 `tl.dot` 累加（避免 Triton 里交错重排），NF4 的 16 项码表用 gather 查表，
每行每 K 块乘一个 absmax，fp32 累加，bf16 输出。嵌套量化的 absmax 在打补丁时一次性解成 fp16 缓存（整模型约 95 MB）。
接入：替换 `bitsandbytes.matmul_4bit`（vLLM 的 `_apply_bnb_4bit` 每次调用都从 `bitsandbytes` 取这个名字；引擎与训练同进程，
`VLLM_ENABLE_V1_MULTIPROCESSING=0`），只接管无梯度、nf4、M ≤ 32、K % 64 == 0 的调用，其余交回原函数。
目标：22 ms → ≤ 10 ms/步；验收看 vLLM decode 的 ms/步和贪心输出一致性。

**P2 lm_head + GRPO loss 融合**：lm_head 冻结，不需要 dW，可按词表分块算 logsumexp 和 dH，不落地 fp32 logits。
先试 TRL 的 `use_liger_loss`（需装 liger-kernel），在 smoke 配置上同 seed 对比 loss、步时、峰值显存；
和 Unsloth 冲突或数值不一致就记录原因，自写 Triton 版另开计划。

**P4 fill / copy / cast 归因**：profile 加 `with_stack` + `record_shapes`，把每个逐元素 kernel 归到发起它的 Python 栈帧，
产出归因表；已知线索是 LoRA 权重 fp32 来回 cast 和反量化缓冲区的零初始化。本项只到归因和修复建议，修复另开计划。

**P3 fp16 累加 GEMM（长期）**：Triton GEMM 用 fp16 累加器、每 256 个 K 提升到 fp32 一次，冲 51 TFLOPS 的峰值。
本计划只做 benchmark 和数值评估（与 fp32 参考的相对误差、梯度余弦），给出 go / no-go；不接入训练。

## 环境隔离

- 代码：新分支 `claude/gpu-kernels`，worktree `.claude/worktrees/gpu-kernels`，从 `claude/project-refactor-model-selection-5f49e3` 切出。
- Python：`~/.venvs/aha-kern`，从验证过的 `~/.venvs/aha` 复制（同版本 torch 2.8 / triton 3.4 / bnb 0.50.2 / vllm 0.10.2），只在这里装 liger-kernel。
- Triton 编译缓存：`TRITON_CACHE_DIR=~/.triton-kern`。
- 跑 GPU 测试前检查 `nvidia-smi`：已用显存 > 2.5 GB 或利用率 > 20% 就拒绝启动（可 `--force`）。
- 所有 benchmark 输出到 `docs/kernels/`，训练产物沿用 `outputs/`（gitignore）。

## 不做的事

- 不改采样数、KL、loss 类型、评测 batch（见 `docs/plan_2026-10-27_next_experiments.md`）。
- 不换 AWQ / Marlin 权重格式（训练要求 bnb 底座，12 GB 放不下两份）。
- 不为 37% "空转" 换系统：trace 证明训练半步是 kernel-bound。

## 验收

| 项 | 指标 | 门槛 |
|---|---|---|
| P0 | bnb gemm_4bit、cuBLAS bf16 GEMM 的基线表 | 带宽 / TFLOPS 百分比与 profile 一致（±15%） |
| P1 | Triton NF4 GEMM，M ∈ {6, 8, 16, 24} | ≥ 2× bnb；误差 ≤ 1.5× bnb 自身误差；vLLM 贪心输出与原版一致率 ≥ 95% 且 decode ms/步下降 ≥ 30% |
| P2 | use_liger_loss 开/关 | 10 步 loss 曲线差异在 1e-3 内，步时和峰值显存有记录，给出结论 |
| P4 | 归因表 | fill / copy / cast 的 ≥ 80% 时间归到具体代码行 |
| P3 | fp16 累加 GEMM | TFLOPS 和相对误差表，go / no-go 结论 |
