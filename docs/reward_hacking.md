# What reward hacking looked like in this GRPO experiment

We trained Qwen2.5-3B-Instruct with GRPO on GSM8K, using 4-bit quantization and
rank-32 LoRA on an RTX 3060 with 12 GB VRAM. The first 500-step run improved its
combined reward while its correctness declined and responses became much shorter.

## The observed mismatch

| Training statistic | First 50 steps | Last 50 steps |
|---|---:|---:|
| Total reward | 1.53 | 2.13 |
| Correctness reward / 2 | 79.0% | 46.7% |
| Mean completion length | 282 tokens | 90 tokens |

These are means over the 500 reward-bearing training-log entries. The final summary
entry is excluded. They are different from held-out GSM8K test accuracy.

![Original 500-step GRPO training curves](../report/img/curves/qwen3b_base_grpo.png)

The faint traces and rolling means are those of the original saved plot. Reward
decomposition shows the integer and XML-count terms increasing while correctness
falls. This is a proxy-versus-task mismatch, consistent with the definition of
[reward hacking](https://arxiv.org/abs/2209.13085). A single wrong answer or a short
response alone would not establish the pattern.

## Three actual held-out outputs

These are complete, verbatim outputs of the final base-reward adapter. The task
descriptions are condensed; indices are the saved zero-based GSM8K test indices.
All three outputs are wrong but contain valid reasoning/answer XML. They illustrate
the failure mode; they are not estimates of how often it occurs.

### Multiplication, example 117

Two pairs of shoes for each of three children, at $60 per pair. Gold: **360**.

```text
<reasoning>
2 * 3 * 60 = 180$
</reasoning>
<answer>
180$
</answer>
```

### A missing condition, example 22

Three customers buy one DVD each; two buy two each; three buy none. Gold: **7**.

```text
<reasoning>
3+2=5
5+0=5
</reasoning>
<answer>
5
</answer>
```

### Almost no reasoning, example 848

There are 30 floral shirts without buttons and 20 white shirts without collars.
How many more floral shirts are there? Gold: **10**.

```text
<reasoning>
0
</reasoning>
<answer>
0
</answer>
```

The full questions and stored predictions are in
[`reward_hacking_examples.json`](reward_hacking_examples.json). A shareable
[curve-and-examples image](../report/img/curves/reward_hacking_explained.png)
keeps the original curve pixels unchanged and adds a short explanation below them.

## Why the reward could mislead the update

GRPO compares the rewards of responses sampled for the same question. When all six
responses receive the same correctness score, correctness contributes no relative
advantage. Differences in the formatting terms can still produce a learning signal,
even though they do not distinguish good mathematical reasoning from bad reasoning.

The initial configuration had no KL penalty to constrain policy drift. Its strict
format checker also required a final newline that generations lacked, so that
reward stayed inactive. The evidence does not mean the model deliberately cheated,
or that DAPO inherently rewards shorter responses. The relationship between short
outputs and declining correctness is an observation from this run.

## Revised configuration and results

[`qwen2.5-3b_3060_fixed.yaml`](../configs/qwen2.5-3b_3060_fixed.yaml) combines:

- Lower format-related bonuses, with correctness still weighted at 2.0.
- No format bonuses when the reasoning block has fewer than 30 words.
- KL regularization with beta 0.04.
- Dr. GRPO loss with constant normalization.
- A repaired strict checker that accepts an omitted final newline.

| Model / checkpoint | GSM8K test accuracy | Mean output tokens |
|---|---:|---:|
| Untouched instruct model | 62.1% | 254 |
| SFT, one epoch | 67.6% | 105 |
| Original GRPO, step 200 | 74.5% | 138 |
| Original GRPO, step 500 | 56.9% | 82 |
| Revised GRPO, step 500 | **77.3%** | 241 |

These results use all 1,319 GSM8K test problems, temperature 0.2, a 512-token
completion limit, and the numeric-normalized matcher in `src/aha/parsing.py`.
They are single-seed comparisons and combine several interventions. They do not
isolate the contribution of each fix. A later no-KL ablation under the revised
reward setup reached 77.2%, so recovery cannot be attributed to KL alone.

The minimum-word gate is a practical guard against empty reasoning, not a verifier
of reasoning quality. Track reward components, held-out correctness, and real
outputs together. Later difficulty-filtering and gate-only experiments did not
significantly improve the revised baseline; see the
[overnight ablations](results_2026-10-27_overnight_queue.md) and
[data-night analysis](results_2026-10-28_data_night.md).

For the original diagnosis and hardware analysis, see
[`results_2026-10-26_grpo_vs_sft.md`](results_2026-10-26_grpo_vs_sft.md).
