#!/usr/bin/env bash
# One-time environment setup for training on an NVIDIA GPU under WSL2 Ubuntu (or native Linux).
#
#   bash scripts/setup_wsl.sh            # creates .venv next to the repo and installs everything
#   source .venv/bin/activate
#   python scripts/check_env.py
#
# Version pins match the Colab notebooks (non-T4 branch): vllm 0.10.2, transformers 4.56.2, trl 0.22.2.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

PY=${PYTHON_VERSION:-3.12}
if [ ! -d .venv ]; then
  echo ">> creating .venv (python $PY)"
  uv venv --python "$PY" .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> installing training stack"
uv pip install --upgrade pip
uv pip install unsloth vllm==0.10.2 bitsandbytes xformers datasets tqdm pyyaml pytest
uv pip install transformers==4.56.2
uv pip install --no-deps trl==0.22.2
uv pip install -e .

echo ">> done. Next:"
echo "   source .venv/bin/activate && python scripts/check_env.py"
echo "   python scripts/train.py --config configs/smoke.yaml"
