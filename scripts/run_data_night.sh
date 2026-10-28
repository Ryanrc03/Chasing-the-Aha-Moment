#!/usr/bin/env bash
# Data night 2026-10-28 (docs/plan_2026-10-28_data_night.md):
#   1. wait until the GPU is free (the user's Windows `rec` process sometimes holds it)
#   2. pre-sample GSM8K train problems 6x with the untouched model (capped)
#   3. run A (difficulty-filtered data) and run B (filtered + gate-only format reward)
#      through scripts/run_queue.sh, full GSM8K eval each, summary vs the fixed baseline
#
#   nohup bash scripts/run_data_night.sh > logs/data_night.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs outputs/presample
PRESAMPLE_OUT=${PRESAMPLE_OUT:-outputs/presample/gsm8k_instruct_g6.jsonl}
PRESAMPLE_MINUTES=${PRESAMPLE_MINUTES:-50}
stage() { echo "=== $(date '+%F %T') $*"; }

wait_for_gpu() {
  local waited=0
  while true; do
    read -r mem util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if [ "$mem" -lt 2500 ] && [ "$util" -lt 20 ]; then return 0; fi
    if [ "$waited" -eq 0 ]; then stage "GPU busy (${mem} MiB, ${util}%), waiting"; fi
    sleep 60; waited=$((waited + 60))
    if [ "$waited" -ge 7200 ]; then stage "GPU still busy after 2 h, giving up"; exit 1; fi
  done
}

wait_for_gpu
stage "presample start (cap ${PRESAMPLE_MINUTES} min)"
timeout $(( (PRESAMPLE_MINUTES + 15) * 60 )) python scripts/presample.py --config configs/qwen2.5-3b_3060_fixed.yaml \
  --out "$PRESAMPLE_OUT" --max-minutes "$PRESAMPLE_MINUTES" > logs/presample.log 2>&1 || stage "presample exited with $?"
grep -E "^PASS-HISTOGRAM|^PRESAMPLE|time cap" logs/presample.log | sed 's/^/    /'
kept=$(python -c "import sys;sys.path.insert(0,'src');from aha.data import load_pass_counts,select_by_pass_count;print(len(select_by_pass_count(load_pass_counts('$PRESAMPLE_OUT'),1,5)))")
total=$(wc -l < "$PRESAMPLE_OUT")
stage "presample done: $total problems sampled, $kept kept (1-5 of 6 right)"
if [ "$kept" -lt 50 ]; then stage "too few problems kept, stopping"; exit 1; fi
[ "$kept" -lt 500 ] && stage "note: fewer than 500 kept, 500 steps will revisit problems"

wait_for_gpu
QUEUE="configs/qwen2.5-3b_3060_fixed_filtered.yaml configs/qwen2.5-3b_3060_fixed_filtered_gate.yaml" \
SWEEP_STEPS="none" \
SUMMARY_OUT=results/data_night_summary.md \
PLOT_DIR=report/img/curves/data_night \
DONE_MARKER=logs/data_night.DONE \
  bash scripts/run_queue.sh
