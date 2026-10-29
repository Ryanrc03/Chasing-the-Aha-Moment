#!/usr/bin/env bash
# GPU stages of the kernel plan, run back to back inside the isolated env. Logs in logs/kq_*.log.
cd "$(dirname "$(readlink -f "$0")")/.."
W=scripts/wsl_env.sh
run() { local name=$1; shift; echo "== $name start $(date +%T)"; bash "$W" "$@" > "logs/kq_$name.log" 2>&1; echo "== $name exit $? $(date +%T)"; }
run decode_base  python scripts/bench_decode.py --out docs/kernels/decode_base.json
run decode_patch python scripts/bench_decode.py --patch --out docs/kernels/decode_patch.json
run decode_cmp   python scripts/bench_decode.py --compare docs/kernels/decode_base.json docs/kernels/decode_patch.json --report docs/kernels/decode.md
run fp16acc      python bench/bench_fp16acc_gemm.py --out docs/kernels/fp16acc_gemm.md
run liger_off    python scripts/train.py --config configs/smoke.yaml --set train.max_steps=10 train.save_steps=10 train.output_dir=outputs/smoke_liger_off train.use_liger_loss=false
run liger_on     python scripts/train.py --config configs/smoke.yaml --set train.max_steps=10 train.save_steps=10 train.output_dir=outputs/smoke_liger_on train.use_liger_loss=true
run attribution  python scripts/profile_step.py --config configs/qwen2.5-3b_3060.yaml --out results/profile_stack --stack --skip-decode
echo "== ALL DONE $(date +%T)"
