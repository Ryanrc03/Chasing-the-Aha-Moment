#!/usr/bin/env bash
# GRPO vs SFT comparison on the RTX 3060, run end to end (several hours):
#   1. SFT baseline            configs/qwen2.5-3b_3060_sft.yaml   -> outputs/qwen3b_sft/lora
#   2. GRPO (base reward)      configs/qwen2.5-3b_3060.yaml       -> outputs/qwen3b_base/lora
#   3. eval instruct / SFT / GRPO on GSM8K test + AIME 2024/2025 -> results/<run>/
#   4. results/comparison.md
#
# Detached use:  nohup bash scripts/run_comparison.sh > logs/comparison.log 2>&1 &
# Set STAGES to a subset (e.g. STAGES="grpo eval") to re-run part of the chain.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STAGES=${STAGES:-"sft grpo eval"}
EVAL_SETS=${EVAL_SETS:-"[gsm8k_test, aime2024, aime2025]"}

stage() { echo "=== $(date '+%F %T') $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has sft; then
  stage "SFT start"
  python scripts/sft.py --config configs/qwen2.5-3b_3060_sft.yaml > logs/sft.log 2>&1 \
    || { echo "SFT FAILED (see logs/sft.log)"; exit 1; }
  stage "SFT done"
fi

if has grpo; then
  stage "GRPO start"
  python scripts/train.py --config configs/qwen2.5-3b_3060.yaml > logs/grpo.log 2>&1 \
    || { echo "GRPO FAILED (see logs/grpo.log)"; exit 1; }
  stage "GRPO done"
fi

if has eval; then
  stage "eval instruct baseline"
  python scripts/eval.py --config configs/qwen2.5-3b_3060.yaml --no-lora \
    --set run_name=qwen3b_instruct "eval.datasets=$EVAL_SETS" > logs/eval_instruct.log 2>&1 \
    || echo "eval instruct FAILED"
  stage "eval SFT"
  python scripts/eval.py --config configs/qwen2.5-3b_3060_sft.yaml \
    --set "eval.datasets=$EVAL_SETS" > logs/eval_sft.log 2>&1 || echo "eval SFT FAILED"
  stage "eval GRPO"
  python scripts/eval.py --config configs/qwen2.5-3b_3060.yaml \
    --set eval.lora_path=outputs/qwen3b_base/lora "eval.datasets=$EVAL_SETS" > logs/eval_grpo.log 2>&1 \
    || echo "eval GRPO FAILED"
  stage "eval done"
fi

python scripts/compare_results.py qwen3b_instruct qwen3b_sft qwen3b_base --out results/comparison.md
stage "ALL DONE"
touch logs/comparison.DONE
