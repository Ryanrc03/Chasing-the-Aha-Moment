#!/usr/bin/env bash
# Evaluate every checkpoint of a GRPO run on a GSM8K-test subset to see when accuracy peaks.
#   bash scripts/sweep_checkpoints.sh outputs/qwen3b_base configs/qwen2.5-3b_3060.yaml 300
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:-outputs/qwen3b_base}
CONFIG=${2:-configs/qwen2.5-3b_3060.yaml}
N=${3:-300}
name=$(basename "$RUN_DIR")
mkdir -p logs
for ckpt in "$RUN_DIR"/checkpoint-*; do
  step=${ckpt##*-}
  echo "=== $(date '+%T') $name step $step"
  python scripts/eval.py --config "$CONFIG" \
    --set run_name="${name}_ckpt${step}" eval.lora_path="$ckpt" eval.max_samples="$N" \
    "eval.datasets=[gsm8k_test]" > "logs/sweep_${name}_${step}.log" 2>&1 || echo "step $step FAILED"
done
python scripts/compare_results.py $(for c in "$RUN_DIR"/checkpoint-*; do echo "${name}_ckpt${c##*-}"; done) \
  --datasets gsm8k_test --out "results/${name}_sweep.md"
echo "=== sweep done"
