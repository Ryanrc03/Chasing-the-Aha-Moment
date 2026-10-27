# Kernel benchmarks

每个 benchmark 一份报告，由 `bench/` 和 `scripts/bench_decode.py` 生成。硬件 RTX 3060 12 GB（GA106）：
显存带宽 360 GB/s；tensor core fp32 累加 25.6 TFLOPS，fp16 累加 51.2 TFLOPS；CUDA core fp32 12.7 TFLOPS。
环境：`~/.venvs/aha-kern`（见 `scripts/wsl_env.sh`）。
