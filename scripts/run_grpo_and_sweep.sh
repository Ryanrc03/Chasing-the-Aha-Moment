#!/usr/bin/env bash
# Train a GRPO config, sweep its checkpoints on a GSM8K subset, evaluate the final adapter on the
# full sets, and tabulate. Detached use:
#   nohup bash scripts/run_grpo_and_sweep.sh configs/qwen2.5-3b_3060_fixed.yaml > logs/fixed.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:?config path}
SUBSET=${2:-300}
mkdir -p logs
run_name=$(python -c "import sys,yaml;sys.path.insert(0,'src');from aha.config import load_config;print(load_config('$CONFIG').run_name)")
out_dir=$(python -c "import sys,yaml;sys.path.insert(0,'src');from aha.config import load_config;print(load_config('$CONFIG').train.output_dir)")
stage() { echo "=== $(date '+%F %T') $*"; }

stage "train $run_name start"
python scripts/train.py --config "$CONFIG" > "logs/train_$run_name.log" 2>&1 \
  || { echo "TRAIN FAILED (logs/train_$run_name.log)"; exit 1; }
stage "train done"

stage "sweep start"
bash scripts/sweep_checkpoints.sh "$out_dir" "$CONFIG" "$SUBSET" > "logs/sweep_$run_name.log" 2>&1 || echo "SWEEP FAILED"
stage "sweep done"

stage "full eval of final adapter"
python scripts/eval.py --config "$CONFIG" --set eval.lora_path="$out_dir/lora" \
  "eval.datasets=[gsm8k_test, aime2024, aime2025]" > "logs/eval_$run_name.log" 2>&1 || echo "EVAL FAILED"
stage "eval done"

python scripts/compare_results.py qwen3b_instruct qwen3b_sft qwen3b_base_step200 qwen3b_base "$run_name" \
  --out "results/comparison_$run_name.md"
stage "ALL DONE"
touch "logs/$run_name.DONE"
