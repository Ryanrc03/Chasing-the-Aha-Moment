#!/usr/bin/env bash
# One-time environment setup for training on an NVIDIA GPU under WSL2 Ubuntu (or native Linux).
#
#   bash scripts/setup_wsl.sh            # creates the venv and installs everything
#   source ~/.venvs/aha/bin/activate     # (or $VENV_DIR if you set it)
#   python scripts/check_env.py
#
# The venv lives on the Linux filesystem (default ~/.venvs/aha) even when the repo is
# under /mnt/<drive>: Python imports from /mnt are several times slower under WSL2.
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
VENV_DIR=${VENV_DIR:-$HOME/.venvs/aha}
if [ ! -d "$VENV_DIR" ]; then
  echo ">> creating venv at $VENV_DIR (python $PY)"
  uv venv --python "$PY" "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo ">> installing training stack"
uv pip install --upgrade pip
uv pip install unsloth vllm==0.10.2 bitsandbytes xformers datasets tqdm pyyaml pytest
uv pip install transformers==4.56.2
uv pip install --no-deps trl==0.22.2
uv pip install -e .

echo ">> done. Next:"
echo "   source $VENV_DIR/bin/activate && python scripts/check_env.py"
echo "   python scripts/train.py --config configs/smoke.yaml"
