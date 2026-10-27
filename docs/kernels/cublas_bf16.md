# cuBLAS GEMM at training shapes, M=2400 (RTX 3060 12 GB (GA106))

tc% against 25.6 TFLOPS (fp32 accumulate); fp32 rows are CUDA-core GEMMs (12.7 TFLOPS peak).

## all shapes

| shape | kernel | ms | TFLOPS | tc% | GB/s | rel_err |
|---|---|---|---|---|---|---|
| q/o_proj 2048x2048 | cuBLAS bf16 | 0.9103 | 22.1156 | 86.4% | 30.8121 | 0.0031 |
| q/o_proj 2048x2048 | cuBLAS fp16 | 0.9073 | 22.1905 | 86.7% | 30.9165 | 0.0004 |
| q/o_proj 2048x2048 | cuBLAS fp32 (cuda cores) | 2.6225 | 7.6770 | 30.0% | 21.3916 | 0.0000 |
| kv_proj 256x2048 | cuBLAS bf16 | 0.1505 | 16.7184 | 65.3% | 80.4354 | 0.0038 |
| kv_proj 256x2048 | cuBLAS fp16 | 0.1495 | 16.8329 | 65.8% | 80.9863 | 0.0005 |
| kv_proj 256x2048 | cuBLAS fp32 (cuda cores) | 0.3553 | 7.0824 | 27.7% | 68.1499 | 0.0000 |
| gate_up 11008x2048 | cuBLAS bf16 | 4.2348 | 25.5536 | 99.8% | 25.4460 | 0.0029 |
| gate_up 11008x2048 | cuBLAS fp16 | 4.3606 | 24.8160 | 96.9% | 24.7116 | 0.0004 |
| gate_up 11008x2048 | cuBLAS fp32 (cuda cores) | 14.2792 | 7.5784 | 29.6% | 15.0930 | 0.0000 |
| down 2048x11008 | cuBLAS bf16 | 4.4780 | 24.1657 | 94.4% | 24.0640 | 0.0029 |
| down 2048x11008 | cuBLAS fp16 | 4.4667 | 24.2267 | 94.6% | 24.1247 | 0.0004 |
| down 2048x11008 | cuBLAS fp32 (cuda cores) | 14.7287 | 7.3471 | 28.7% | 14.6323 | 0.0000 |
| lora_A 32x2048 | cuBLAS bf16 | 0.0492 | 6.4000 | 25.0% | 205.7917 | 0.0029 |
| lora_A 32x2048 | cuBLAS fp16 | 0.0523 | 6.0198 | 23.5% | 193.5677 | 0.0004 |
| lora_A 32x2048 | cuBLAS fp32 (cuda cores) | 0.0973 | 3.2337 | 12.6% | 207.9579 | 0.0000 |
| lora_B 2048x32 | cuBLAS bf16 | 0.0369 | 8.5333 | 33.3% | 274.3889 | 0.0029 |
| lora_B 2048x32 | cuBLAS fp16 | 0.0399 | 7.8769 | 30.8% | 253.2821 | 0.0004 |
| lora_B 2048x32 | cuBLAS fp32 (cuda cores) | 0.0829 | 3.7926 | 14.8% | 243.9012 | 0.0000 |
| lm_head 151936x2048 | cuBLAS bf16 | 57.4449 | 26.0004 | 101.6% | 23.7002 | 0.0029 |
| lm_head 151936x2048 | cuBLAS fp16 | 57.1960 | 26.1136 | 102.0% | 23.8033 | 0.0004 |
| lm_head 151936x2048 | cuBLAS fp32 (cuda cores) | 205.5168 | 7.2675 | 28.4% | 13.2491 | 0.0000 |

## 结论（基线，2026-10-27）

- 大矩阵（gate_up / down / lm_head）的 cuBLAS bf16 达到 25.6 TFLOPS 的 94%–102%：fp32 累加峰值的假设成立，profile 里 14.4 TFLOPS 的 GEMM 平均是被小形状和 LoRA 分支拉低的，不是 cuBLAS 本身低效。
- cuBLAS fp16 和 bf16 一样快：PyTorch 走的 cuBLAS 路径不用 fp16 累加，P3 只能靠自写 kernel 验证。
- LoRA 的两个 GEMM（K=32 或 N=32）是带宽型：200–270 GB/s，tensor core 只用到 25%–33%；值得融合而不是调 tile。
- fp32 CUDA core 路径是 7.3–7.7 TFLOPS（峰值 12.7 的 60%），trace 里 LoRA 的 sgemm 就是这条路，比 bf16 慢 3 倍。
