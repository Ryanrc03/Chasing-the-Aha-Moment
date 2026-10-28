# vLLM decode: bitsandbytes kernel vs Triton NF4 GEMM

Qwen2.5-3B-Instruct bnb-4bit, 256 decode steps per run, median of 3.

| sequences | base ms/step | patch ms/step | speed-up | base tok/s | patch tok/s |
|---|---|---|---|---|---|
| 6 | 29.8 | 24.8 | 1.20x | 202 | 242 |
| 24 | 61.4 | 53.5 | 1.15x | 391 | 449 |

Greedy agreement on 20 GSM8K prompts: 6/20 identical token sequences (30%); first divergence at token index [15, 30, 30, 30, 50, 67, 88, 89, 92, 107, 121, 150, 151, 212] for the rest.

## Notes (2026-10-27)

- The speed-up matches the microbenchmark: per-layer 4-bit GEMM time 0.55 -> 0.44 ms at M=6, i.e. about 4 ms of the 22 ms
  that bnb spends per decode step; measured end to end 29.8 -> 24.8 ms (-5.0 ms). The 24-sequence row gains less because
  M=24 calls are routed back to bitsandbytes (the Triton tile is slower there).
- Greedy outputs diverge late (first differing token at index 15-212). Final answers agree on 14/20 and accuracy is 12/20
  (bnb) vs 9/20 (Triton) with the 256-token cap; 20 prompts cannot separate kernel noise from a real loss, so the
  paired full GSM8K evaluation below is the acceptance test (`scripts/run_nf4_accuracy.sh`).

## Acceptance: paired GSM8K test, fixed-run LoRA (2026-10-27, `scripts/run_nf4_accuracy.sh`)

| | bitsandbytes decode | Triton NF4 decode |
|---|---|---|
| accuracy (1,319) | 0.7733 | 0.7642 |
| eval wall time | 1,970 s | 1,878 s (-4.7%) |
| avg completion tokens | 242.6 | 241.6 |

Paired: 50 questions right only under bnb, 38 right only under Triton; exact McNemar p = 0.24. The -0.9 point gap is not
significant. Eval samples at temperature 0.2, so any change in numerics re-rolls some samples; the run-to-run noise floor
of bnb vs bnb has not been measured and is the remaining check before calling the kernel neutral.
Eval wall time drops far less than the 1.20x decode step because eval runs batches of 8 with prefill, LoRA and early-finishing
sequences, and only the M <= 16 decode GEMMs are routed to the new kernel.
