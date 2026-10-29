# Original Colab notebooks

These four notebooks preserve the original Llama 3.1-8B / T4 course experiments,
including their saved outputs. They have been moved here without changing their contents.

- `Llama3_1_(8B)_GRPO_250.ipynb`: 250-step training notebook.
- `Llama3_1_(8B)_GRPO_250_eval.ipynb`: training and GSM8K / AIME evaluation.
- `Llama3.1_(8B)-GRPO.ipynb`: earlier training version, formerly in `nb/`.
- `Llama3.1_(8B)_T4-GRPO.ipynb`: T4 variant, formerly in `nb/`.

For the current Qwen2.5-3B experiments, use the Python package in
[`src/aha/`](../../src/aha/), the CLI entry points in [`scripts/`](../../scripts/),
and the YAML files in [`configs/`](../../configs/). See the root
[README](../../README.md) for setup and results. The original course report remains
in [`report/`](../../report/); its Llama results are separate from the newer Qwen runs.
