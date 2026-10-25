# Chasing the "Aha Moment"

**A Review and Implementation of Reinforcement Learning in the Post-Training of LLMs**

Course project (COMP 514, Rice University) by Ryan Li, Serena You, Yuanyuan Zhu, and Diwesh Chaurasia.

We review RL-based post-training methods for LLMs (SFT, RLHF, DPO, GRPO) and train
Llama 3.1-8B-Instruct with **GRPO** on GSM8K under a single-GPU budget (Colab T4, 4-bit + LoRA),
using handcrafted reward functions, including a length-based term that rewards longer reasoning
when the answer is wrong, to probe whether reward shaping can elicit "Aha Moment"-like behavior.

## Results

| Metric (GSM8K test, 1,319 problems) | GRPO (base reward) | GRPO + reward shaping |
|---|---|---|
| Accuracy | 0.8256 | **0.8500** |
| XML parse rate | 1.0000 | 0.9980 |
| Full `<reasoning>/<answer>` format | 0.0000 | **0.2300** |

On the combined AIME 2024 + 2025 set (60 problems) both models score 0, while format
consistency still improves with reward shaping. Our reading: at the 8B scale GRPO makes
reasoning more structured and consistent but does not add new mathematical capability.
See the report for details.

## Repository layout

```
.
├── Llama3_1_(8B)_GRPO_250.ipynb           # GRPO training notebook (250 steps, Colab T4)
├── Llama3_1_(8B)_GRPO_250_eval.ipynb      # Same training run + GSM8K / AIME evaluation cells
├── nb/
│   ├── Llama3.1_(8B)-GRPO.ipynb           # Earlier training notebook
│   └── Llama3.1_(8B)_T4-GRPO.ipynb        # T4-specific variant
└── report/
    ├── main.tex                           # Report source (NeurIPS 2024 style)
    ├── references.bib
    ├── neurips_2024.sty
    ├── img/
    └── report.pdf                         # Compiled report
```

## Running the training notebook

The notebooks are written for Google Colab with a T4 (15 GB). They install
`unsloth`, `vllm`, `trl==0.22.2`, and `transformers==4.56.2`, load
`unsloth/meta-Llama-3.1-8B-Instruct` in 4-bit with a rank-32 LoRA, and run 250 GRPO steps
with 6 generations per prompt (max prompt 256 tokens, max completion 768 tokens).

Hardware notes:

- The 8B configuration as written needs roughly 13 GB of GPU memory because vLLM is co-located
  with the trainer. It fits a 15 GB T4 but **not** a 12 GB card such as an RTX 3060 without
  lowering `gpu_memory_utilization`, `max_seq_length`, `num_generations`, or switching to a 3B model.
- Unsloth + vLLM require Linux (or WSL2 on Windows).

## Building the report

```bash
cd report
tectonic main.tex        # or: pdflatex + bibtex + pdflatex x2
```

## License

MIT, see [LICENSE](LICENSE).
