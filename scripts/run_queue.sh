#!/usr/bin/env bash
# Run several GRPO configs back to back: train, sweep a few checkpoints on 300 GSM8K test
# problems, evaluate the final adapter on the full test set, then summarise everything.
# One run failing (OOM, timeout) does not stop the next.
#
#   nohup bash scripts/run_queue.sh > logs/queue.log 2>&1 &
#   QUEUE="configs/a.yaml configs/b.yaml" bash scripts/run_queue.sh
#
# Plan and reading guide: docs/plan_2026-10-27_next_experiments.md
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results

QUEUE=${QUEUE:-"configs/qwen2.5-3b_3060_fixed_lr2e5.yaml configs/qwen2.5-3b_3060_fixed_noscale.yaml configs/qwen2.5-3b_3060_fixed_nokl.yaml configs/qwen2.5-3b_3060_fixed_g8.yaml"}
TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-14400}   # 4 h
SWEEP_STEPS=${SWEEP_STEPS:-"200 300 400"}
FULL_TIMEOUT=${FULL_TIMEOUT:-4500}      # 75 min for 1,319 problems

stage() { echo "=== $(date '+%F %T') $*"; }
field() { python -c "import sys;sys.path.insert(0,'src');from aha.config import load_config;c=load_config('$1');print($2)"; }

runs=()
for cfg in $QUEUE; do
  name=$(field "$cfg" "c.run_name")
  out=$(field "$cfg" "c.train.output_dir")
  runs+=("$name")

  stage "$name train start ($cfg)"
  if timeout "$TRAIN_TIMEOUT" python scripts/train.py --config "$cfg" > "logs/train_$name.log" 2>&1; then
    stage "$name train done"
  else
    code=$?
    stage "$name TRAIN FAILED (exit $code; 124 = timeout); tail of logs/train_$name.log:"
    grep -E "Error|error|Killed|out of memory" "logs/train_$name.log" | tail -3
    # a partial run still has checkpoints worth sweeping
  fi

  stage "$name sweep (steps $SWEEP_STEPS)"
  STEPS="$SWEEP_STEPS" bash scripts/sweep_checkpoints.sh "$out" "$cfg" 300 > "logs/sweep_$name.log" 2>&1
  grep -E "FAILED" "logs/sweep_$name.log" | sed "s/^/    /"

  if [ -f "$out/lora/adapter_model.safetensors" ]; then
    stage "$name full GSM8K eval of the final adapter"
    timeout "$FULL_TIMEOUT" python scripts/eval.py --config "$cfg" --set eval.lora_path="$out/lora" \
      "eval.datasets=[gsm8k_test]" > "logs/eval_$name.log" 2>&1 || stage "$name EVAL FAILED (exit $?)"
    grep -E "^ +accuracy" "logs/eval_$name.log" | sed "s/^/    /"
  else
    stage "$name has no final adapter, skipping full eval"
  fi
done

stage "summary"
{
  echo "# Overnight queue summary ($(date '+%F %T'))"
  echo
  echo "Baseline: qwen3b_fixed, GSM8K test 0.7726. See docs/plan_2026-10-27_next_experiments.md."
  echo
  python scripts/compare_results.py qwen3b_fixed "${runs[@]}" --datasets gsm8k_test
  echo
  python scripts/analyze_hacking.py qwen3b_fixed "${runs[@]}"
} > results/queue_summary.md 2>&1
python scripts/plot_curves.py outputs/qwen3b_fixed $(for r in "${runs[@]}"; do [ -d "outputs/$r" ] && echo "outputs/$r"; done) \
  --compare rewards/correctness_reward_func/mean --out report/img/curves/queue > /dev/null 2>&1 || true
python scripts/plot_curves.py outputs/qwen3b_fixed $(for r in "${runs[@]}"; do [ -d "outputs/$r" ] && echo "outputs/$r"; done) \
  --compare completion_length --out report/img/curves/queue > /dev/null 2>&1 || true
stage "ALL DONE"
touch logs/queue.DONE
