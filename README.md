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
├── src/aha/                 # importable package (no GPU needed to import or test)
│   ├── config.py            #   YAML configs with _base_ inheritance + dotted overrides
│   ├── prompts.py           #   system prompt / XML format
│   ├── parsing.py           #   answer extraction, numeric normalisation
│   ├── data.py              #   GSM8K / MATH train; GSM8K-test / AIME24 / AIME25 / MATH-500 eval
│   ├── rewards.py           #   5 base rewards + length-based reward shaping (R_len)
│   ├── model.py             #   Unsloth 4-bit + LoRA (+ vLLM) loading
│   ├── train.py             #   TRL GRPOTrainer wiring
│   └── evaluate.py          #   batched vLLM eval -> metrics.json / predictions.jsonl
├── scripts/
│   ├── train.py             # python scripts/train.py --config configs/<name>.yaml [--set k=v ...]
│   ├── eval.py              # python scripts/eval.py  --config configs/<name>.yaml [--set k=v ...]
│   ├── check_env.py         # GPU + package versions
│   └── setup_wsl.sh         # one-shot venv + pinned install for WSL2 / Linux
├── configs/
│   ├── base.yaml            # shared defaults (report Table 1 hyper-parameters)
│   ├── qwen2.5-3b_3060.yaml         # RTX 3060 12 GB, base reward
│   ├── qwen2.5-3b_3060_shaped.yaml  # RTX 3060 12 GB, + length reward shaping
│   ├── llama3.2-3b_3060.yaml        # same-family 3B alternative
│   ├── llama3.1-8b_t4.yaml          # original Colab/T4 run
│   └── smoke.yaml                   # 2-step end-to-end check
├── tests/                   # CPU-only unit tests (pytest)
├── docs/
│   └── model_and_data_selection.md  # 3060 model / data choices (中文)
├── notebooks/colab/         # original Colab notebooks, kept for reference
└── report/                  # NeurIPS-style report source + PDF
```

## Quick start (WSL2 Ubuntu or Linux with an NVIDIA GPU)

```bash
bash scripts/setup_wsl.sh          # creates .venv, installs unsloth / vllm / trl (pinned)
source ~/.venvs/aha/bin/activate
python scripts/check_env.py

python scripts/train.py --config configs/smoke.yaml           # 2 steps, verifies the pipeline
python scripts/train.py --config configs/qwen2.5-3b_3060.yaml  # base reward
python scripts/train.py --config configs/qwen2.5-3b_3060_shaped.yaml  # + length reward

# evaluate the instruct baseline, then a trained adapter
python scripts/eval.py --config configs/qwen2.5-3b_3060.yaml --set eval.lora_path=null
python scripts/eval.py --config configs/qwen2.5-3b_3060.yaml \
    --set eval.lora_path=outputs/qwen3b_base/lora "eval.datasets=[gsm8k_test, math500, aime2024, aime2025]"
```

Any config key can be overridden on the command line with `--set section.key=value`
(values are parsed as YAML). Results land in `results/<run_name>/<dataset>/`.

Unit tests run anywhere, no GPU or model download:

```bash
pip install -e ".[dev]" && pytest
```

## Hardware notes

- **RTX 3060 (12 GB):** use the 3B configs. The 8B configuration needs roughly 13 GB because
  vLLM is co-located with the trainer and does not fit. See
  [docs/model_and_data_selection.md](docs/model_and_data_selection.md) for the memory budget,
  model comparison and recommended experiment order.
- **Colab T4 (15 GB):** `configs/llama3.1-8b_t4.yaml` reproduces the original notebooks
  (`unsloth/meta-Llama-3.1-8B-Instruct`, 4-bit, rank-32 LoRA, 250 steps, 6 generations,
  256/768 prompt/completion tokens).
- Unsloth + vLLM require Linux (WSL2 on Windows). Pinned versions: `vllm==0.10.2`,
  `transformers==4.56.2`, `trl==0.22.2`. Verified on the 3060 under WSL2 with Unsloth 2026.10.8,
  torch 2.8.0+cu128: the 3B configs run at about 17-21 s per step with 6 generations (standard gradient
  checkpointing; Unsloth's CPU-offload variant is ~35% slower under WSL2), so 500 steps take
  roughly 2.5-3 hours. The model download is 2.3 GB.
- vLLM standby mode is off by default (`UNSLOTH_VLLM_STANDBY=0`): current Unsloth refuses it on
  vLLM 0.10.x. Upgrade to `vllm>=0.11.2` before turning it on.

## Building the report

```bash
cd report
tectonic main.tex        # or: pdflatex + bibtex + pdflatex x2
```

## License

MIT, see [LICENSE](LICENSE).
