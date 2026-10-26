# Refactor notebooks into a Python package (design)

Date: 2026-10-25

## Goal

Turn the four Colab notebooks into an importable package plus CLI scripts so that
GRPO experiments can be iterated locally on an RTX 3060 (12 GB, WSL2 Ubuntu).
Keep the original Llama-3.1-8B / T4 run reproducible from a config file.

## Layout

```
src/aha/            importable package (no GPU needed to import)
  config.py         dataclass configs, YAML loading, `_base_` inheritance, dotted overrides
  prompts.py        SYSTEM_PROMPT and XML format constants
  parsing.py        answer extraction and relaxed numeric comparison
  data.py           GSM8K train loader; GSM8K-test / AIME24 / AIME25 / MATH-500 eval loaders
  rewards.py        the five base rewards + length-based reward (alpha, beta, L_max)
  model.py          Unsloth model/LoRA loading
  train.py          build GRPOConfig + GRPOTrainer, run, save LoRA
  evaluate.py       batched vLLM generation, metrics, JSON/JSONL outputs
scripts/            train.py, eval.py, check_env.py, setup_wsl.sh
configs/            base.yaml + one file per (model, hardware) + reward variants
tests/              CPU-only unit tests (parsing, rewards, config, data formatting)
notebooks/colab/    original notebooks, kept for reference
docs/               model/data selection notes for the 3060
```

## Decisions

- **Primary 3060 target:** Qwen2.5-3B-Instruct, 4-bit + LoRA r=32, vLLM co-located at
  `gpu_memory_utilization=0.55`, `max_seq_length=1024`, 6 generations. Llama-3.2-3B-Instruct
  is the family-consistent alternative. The 8B config stays for Colab/T4 reproduction.
- **Data:** GSM8K train for RL. Eval on GSM8K test, AIME 2024, AIME 2025 (as in the report)
  plus MATH-500 as a medium-difficulty bridge between the two.
- **Rewards:** the length-based term from the report is implemented as a config-gated reward
  (`reward.length.enabled`), so "base" vs "shaped" is a one-line config change.
- **Correctness match:** `exact` (as in the notebooks) or `relaxed` (numeric normalisation,
  as in the eval cell); configurable, default `exact` to reproduce.
- Heavy imports (unsloth, trl, vllm, datasets) live inside functions so the package and
  tests import on Windows without a GPU.

## Testing

`pytest tests/` runs on CPU with no model downloads. GPU runs are smoke-tested with
`scripts/train.py --set train.max_steps=2` inside WSL2.
