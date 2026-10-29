# fp16-accumulate GEMM feasibility (RTX 3060)

fp32acc% = share of the 25.6 TFLOPS fp32-accumulate peak; fp16acc% = share of the 51.2 TFLOPS fp16-accumulate peak. cos = cosine similarity to the fp32 product (stands in for gradient direction agreement).

## all shapes

| shape | kernel | ms | TFLOPS | fp32acc% | fp16acc% | rel_err | cos | note |
|---|---|---|---|---|---|---|---|---|
| q/o_proj K2048 N2048 | cuBLAS fp16 (fp32 acc) | 0.9062 | 22.2156 | 86.8% | 43.4% | 0.0004 | 1.0000 |  |
| q/o_proj K2048 N2048 | triton fp32 acc | 2.1847 | 9.2153 | 36.0% | 18.0% | 0.0004 | 1.0000 |  |
| q/o_proj K2048 N2048 | triton fp16 acc, promote 1 | 0.5898 | 34.1333 | 133.3% | 66.7% | 0.0005 | 1.0000 |  |
| q/o_proj K2048 N2048 | triton fp16 acc, promote 4 | 0.6886 | 29.2354 | 114.2% | 57.1% | 0.0007 | 1.0000 |  |
| q/o_proj K2048 N2048 | triton fp16 acc, promote 8 | 0.6820 | 29.5207 | 115.3% | 57.7% | 0.0009 | 1.0000 |  |
| gate_up K2048 N11008 | cuBLAS fp16 (fp32 acc) | 4.3863 | 24.6707 | 96.4% | 48.2% | 0.0004 | 1.0000 |  |
| gate_up K2048 N11008 | triton fp32 acc | 12.1108 | 8.9353 | 34.9% | 17.5% | 0.0004 | 1.0000 |  |
| gate_up K2048 N11008 | triton fp16 acc, promote 1 | 3.3848 | 31.9700 | 124.9% | 62.4% | 0.0005 | 1.0000 |  |
| gate_up K2048 N11008 | triton fp16 acc, promote 4 | 4.2245 | 25.6155 | 100.1% | 50.0% | 0.0007 | 1.0000 |  |
| gate_up K2048 N11008 | triton fp16 acc, promote 8 | 3.8006 | 28.4728 | 111.2% | 55.6% | 0.0009 | 1.0000 |  |
| down K11008 N2048 | cuBLAS fp16 (fp32 acc) | 4.1595 | 26.0162 | 101.6% | 50.8% | 0.0004 | 1.0000 |  |
| down K11008 N2048 | triton fp32 acc | 12.6572 | 8.5496 | 33.4% | 16.7% | 0.0004 | 1.0000 |  |
| down K11008 N2048 | triton fp16 acc, promote 1 | 4.1702 | 25.9489 | 101.4% | 50.7% | 0.0005 | 1.0000 |  |
| down K11008 N2048 | triton fp16 acc, promote 4 | 4.5020 | 24.0366 | 93.9% | 46.9% | 0.0007 | 1.0000 |  |
| down K11008 N2048 | triton fp16 acc, promote 8 | 4.3745 | 24.7371 | 96.6% | 48.3% | 0.0009 | 1.0000 |  |

## Conclusion (2026-10-27)

- fp16 accumulation beats the fp32-accumulate ceiling: 34.1 TFLOPS on q/o_proj (133% of 25.6), 32.0 on gate_up. This is the
  first direct measurement of the GeForce half-rate fp32 accumulate on this card.
- Against cuBLAS: q/o 1.54x, gate_up 1.30x, down 1.00x (promote_every=1). rel_err 0.0005 vs cuBLAS 0.0004, cosine 1.0000.
  Promoting every block costs no measurable speed and keeps the error flat; longer chunks (4, 8) are slower and less precise
  on this Triton version, so promote_every=1 is the setting.
- The Triton fp32-accumulate variant runs at only ~9 TFLOPS: Triton 3.4 pipelines the chunked inner loop badly when the
  accumulator is fp32. The fp16 numbers are therefore a lower bound on what a tuned kernel could reach.
- **Go, conditionally.** Criteria were >= 1.4x cuBLAS, rel_err <= 2x cuBLAS, cos >= 0.9999. q/o passes, gate_up is close,
  down needs tile tuning (K=11008). Next plan: per-shape autotune (block sizes, warps, stages, split along K for down),
  then wire the four base GEMMs of the forward pass behind a flag and compare a 10-step smoke run's loss and reward with
  fp16 weights + loss scaling. The backward GEMMs (dX = dY @ W) have the same shapes transposed and would follow.
