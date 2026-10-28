# use_liger_loss (TRL 0.22.2 + Unsloth 2026.10.8)

liger-kernel 0.8.5, installed only in the isolated `~/.venvs/aha-kern`. Smoke config, same seed, 10 steps each.

| step | loss off | loss on | diff | len off | len on | peak GB off | peak GB on |
|---|---|---|---|---|---|---|---|
| 1 | 0.3612 | 0.3612 | 0 | 440 | 440 | 7.67 | 7.67 |
| 2 | 0.2645 | 0.2645 | 0 | 378 | 378 | 7.67 | 7.67 |
| 3 | 0.1556 | 0.1363 | 1.9e-02 | 254 | 250 | 7.67 | 7.67 |
| 5 | 0.0752 | 0.2819 | 2.1e-01 | 249 | 278 | 7.67 | 7.67 |
| 10 | 0.0353 | 0.1047 | 6.9e-02 | 284 | 268 | 7.67 | 7.67 |

- Median step time: off 11.6 s, on 11.3 s (within noise).
- Peak GPU memory: identical at 7.67 GB on every step.

## Conclusion: the switch is a no-op under Unsloth

`unsloth_compiled_cache/UnslothGRPOTrainer.py` builds `self.liger_grpo_loss` when the flag is set, but Unsloth's own
`compute_loss` always goes through `grpo_accumulated_loss`, which already computes log-probs chunk by chunk
(`chunked_selective_log_softmax`). The Liger method is never called. The two runs agree exactly for two steps, then split
because sampling is not bitwise reproducible from run to run: completion lengths differ from step 3 on.

Fusing lm_head with the loss would need a kernel of our own, plugged into Unsloth's `grpo_accumulated_loss`. Expected gain is
small, because Unsloth already avoids materialising the full fp32 logits. The profile's softmax/memcpy share came from
`profile_step.py`'s plain `log_softmax`, not from the trainer. Not worth doing before P1/P3 follow-ups.
