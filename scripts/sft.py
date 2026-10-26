#!/usr/bin/env python
"""Train the SFT baseline LoRA from a YAML config.

    python scripts/sft.py --config configs/qwen2.5-3b_3060_sft.yaml
    python scripts/sft.py --config configs/qwen2.5-3b_3060_sft.yaml --set sft.max_steps=10

Evaluate the result with scripts/eval.py and ``--set eval.lora_path=<sft.output_dir>/lora``.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")

try:
    import unsloth  # noqa: F401  (must precede transformers/trl imports)
except ImportError:
    print("unsloth is not installed; run scripts/setup_wsl.sh inside WSL2/Linux.", file=sys.stderr)
    raise

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aha.config import load_config  # noqa: E402
from aha.sft import train_sft  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    print(f"run_name={cfg.run_name} model={cfg.model.name} epochs={cfg.sft.num_train_epochs} "
          f"max_steps={cfg.sft.max_steps} lr={cfg.sft.learning_rate}")
    train_sft(cfg)


if __name__ == "__main__":
    main()
