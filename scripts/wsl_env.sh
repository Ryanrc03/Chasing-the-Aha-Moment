#!/usr/bin/env bash
# Run a command inside the isolated kernel-work environment (WSL side).
#   wsl -d Ubuntu -- bash /mnt/d/.../gpu-kernels/scripts/wsl_env.sh python -m pytest tests -v
set -euo pipefail
source ~/.venvs/aha-kern/bin/activate
export TRITON_CACHE_DIR="$HOME/.triton-kern"
export UNSLOTH_VLLM_STANDBY=0
cd "$(dirname "$(readlink -f "$0")")/.."
exec "$@"
