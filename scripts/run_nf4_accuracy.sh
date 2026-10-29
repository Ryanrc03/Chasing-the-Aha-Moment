#!/usr/bin/env bash
# Paired GSM8K test eval of the fixed-run LoRA: bitsandbytes decode vs Triton NF4 decode, identical settings.
cd "$(dirname "$(readlink -f "$0")")/.."
W=scripts/wsl_env.sh
LORA=/mnt/d/sideProjects/Chasing-the-Aha-Moment/.claude/worktrees/project-refactor-model-selection-5f49e3/outputs/qwen3b_fixed/lora
COMMON="eval.datasets=[gsm8k_test] eval.lora_path=$LORA eval.results_dir=results/nf4_acc"
echo "== bnb start $(date +%T)"
timeout 3000 bash $W python scripts/eval.py --config configs/qwen2.5-3b_3060.yaml --set run_name=bnb $COMMON > logs/acc_bnb.log 2>&1
echo "== bnb exit $? $(date +%T)"
echo "== triton start $(date +%T)"
AHA_TRITON_NF4=1 timeout 3000 bash $W python scripts/eval.py --config configs/qwen2.5-3b_3060.yaml --set run_name=triton $COMMON > logs/acc_triton.log 2>&1
echo "== triton exit $? $(date +%T)"
echo "== ALL DONE"
