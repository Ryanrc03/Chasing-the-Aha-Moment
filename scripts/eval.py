#!/usr/bin/env python
"""Evaluate a base model, or base + trained LoRA, on GSM8K / AIME / MATH-500.

    # instruct baseline
    python scripts/eval.py --config configs/qwen2.5-3b_3060.yaml --set eval.lora_path=null

    # trained adapter, all eval sets
    python scripts/eval.py --config configs/qwen2.5-3b_3060.yaml \
        --set eval.lora_path=outputs/qwen3b_base/lora "eval.datasets=[gsm8k_test, aime2024, aime2025]"
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

try:
    import unsloth  # noqa: F401
except ImportError:
    print("unsloth is not installed; run scripts/setup_wsl.sh inside WSL2/Linux.", file=sys.stderr)
    raise

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aha.config import load_config  # noqa: E402
from aha.evaluate import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if cfg.eval.lora_path is None:
        # default to the adapter this config's training run would have produced, if it exists
        candidate = Path(cfg.lora_save_dir)
        if candidate.exists():
            cfg.eval.lora_path = str(candidate)
    print(f"run_name={cfg.run_name} model={cfg.model.name} lora={cfg.eval.lora_path} "
          f"datasets={cfg.eval.datasets}")
    evaluate(cfg)


if __name__ == "__main__":
    main()
