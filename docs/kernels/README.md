# Kernel benchmarks

每个 benchmark 一份报告，由 `bench/` 和 `scripts/bench_decode.py` 生成。硬件 RTX 3060 12 GB（GA106）：
显存带宽 360 GB/s；tensor core fp32 累加 25.6 TFLOPS，fp16 累加 51.2 TFLOPS；CUDA core fp32 12.7 TFLOPS。
环境：`~/.venvs/aha-kern`（见 `scripts/wsl_env.sh`）。

## 报告

- `gemm_4bit.md`：decode 形状的 4-bit GEMM，bnb / cuBLAS / Triton 对比，以及 Triton 设计探索的结论。
- `cublas_bf16.md`：训练形状的 cuBLAS 基线，证实 fp32 累加峰值为 25.6 TFLOPS。
- `decode.md`：vLLM decode 端到端 A/B，以及 GSM8K 全量配对评测。
- `fp16acc_gemm.md`：fp16 累加 GEMM 的速度和误差，go / no-go。
- `liger_loss.md`：`use_liger_loss` 在 Unsloth 下无效的证据。
- `attribution.md`：训练半步逐元素 kernel 的代码归因。
- 汇总：`docs/results_2026-10-27_kernels.md`。
