#!/usr/bin/env python
"""Train a GRPO LoRA from a YAML config.

    python scripts/train.py --config configs/qwen2.5-3b_3060.yaml
    python scripts/train.py --config configs/qwen2.5-3b_3060.yaml --set train.max_steps=10

Dotted ``--set`` overrides are parsed as YAML (``true``, ``0.5``, ``[a, b]`` all work).
"""

import argparse
import os
import sys
from pathlib import Path

# vLLM standby mode (frees vLLM memory between generation phases) is refused by current
# Unsloth releases on the pinned vllm 0.10.2 (std::bad_alloc); enable it only on vllm>=0.11.2:
#   UNSLOTH_VLLM_STANDBY=1 python scripts/train.py ...
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")

# Opt-in: route decode-sized 4-bit matmuls to the Triton NF4 GEMM (docs/kernels/gemm_4bit.md).
# Must run before vLLM builds its engine, i.e. before the model is loaded.
if os.environ.get("AHA_TRITON_NF4", "0") == "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from aha.kernels.nf4_gemm import patch_bitsandbytes

    patch_bitsandbytes()
    print("AHA_TRITON_NF4=1: bitsandbytes.matmul_4bit routed to the Triton NF4 GEMM")

try:
    import unsloth  # noqa: F401  (must precede transformers/trl imports)
except ImportError:
    print("unsloth is not installed; run scripts/setup_wsl.sh inside WSL2/Linux.", file=sys.stderr)
    raise

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aha.config import load_config  # noqa: E402
from aha.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    print(f"run_name={cfg.run_name} model={cfg.model.name} steps={cfg.train.max_steps} "
          f"gens={cfg.train.num_generations} length_reward={cfg.reward.length.enabled}")
    train(cfg)


if __name__ == "__main__":
    main()
