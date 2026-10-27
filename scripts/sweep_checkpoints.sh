#!/usr/bin/env bash
# Evaluate checkpoints of a GRPO run on a GSM8K-test subset to see when accuracy peaks.
#   bash scripts/sweep_checkpoints.sh outputs/qwen3b_base configs/qwen2.5-3b_3060.yaml 300
# Optional env:
#   STEPS="200 300 400"   only these checkpoints (default: every checkpoint-* in the run)
#   EVAL_TIMEOUT=1800     seconds per checkpoint before the eval is killed (vLLM occasionally hangs)
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:-outputs/qwen3b_base}
CONFIG=${2:-configs/qwen2.5-3b_3060.yaml}
N=${3:-300}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-1800}
name=$(basename "$RUN_DIR")
mkdir -p logs

if [ -n "${STEPS:-}" ]; then
  ckpts=$(for s in $STEPS; do [ -d "$RUN_DIR/checkpoint-$s" ] && echo "$RUN_DIR/checkpoint-$s"; done)
else
  ckpts=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
fi

done_names=()
for ckpt in $ckpts; do
  step=${ckpt##*-}
  echo "=== $(date '+%T') $name step $step"
  timeout "$EVAL_TIMEOUT" python scripts/eval.py --config "$CONFIG" \
    --set run_name="${name}_ckpt${step}" eval.lora_path="$ckpt" eval.max_samples="$N" \
    "eval.datasets=[gsm8k_test]" > "logs/sweep_${name}_${step}.log" 2>&1 \
    && done_names+=("${name}_ckpt${step}") || echo "step $step FAILED (exit $?)"
done
if [ ${#done_names[@]} -gt 0 ]; then
  python scripts/compare_results.py "${done_names[@]}" --datasets gsm8k_test --out "results/${name}_sweep.md"
fi
echo "=== sweep done"
