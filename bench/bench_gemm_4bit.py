#!/usr/bin/env python
"""Decode-shaped 4-bit GEMM: y[M, N] = x[M, K] @ W_nf4[N, K]^T for the Qwen2.5-3B projections.

Candidates: bitsandbytes' fused kernel (what vLLM calls), bnb dequant + cuBLAS (its fallback),
cuBLAS on a bf16 copy of the weight (the 16-bit bound), and our Triton kernel when present.

    python bench/bench_gemm_4bit.py --out docs/kernels/gemm_4bit.md
    python bench/bench_gemm_4bit.py --triton --out docs/kernels/gemm_4bit.md
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.harness import assert_gpu_free, rel_fro_error, render_table, time_ms, write_report  # noqa: E402

# (name, N, K) of the Qwen2.5-3B linear layers (hidden 2048, intermediate 11008, 2 kv heads x 128)
QWEN_SHAPES = [("q_proj", 2048, 2048), ("kv_proj", 256, 2048), ("gate_up", 11008, 2048), ("down", 2048, 11008)]
MS = [6, 8, 16, 24, 32]
ROTATE_BYTES = 48 * 2**20  # copies of each weight so consecutive calls miss L2


def make_weights(N, K, copies, device="cuda"):
    import bitsandbytes.functional as F
    import torch

    from aha.kernels.nf4 import dequant_nf4_reference, resolve_absmax

    W = (torch.randn(N, K, device=device) * 0.02).to(torch.bfloat16)
    packs = []
    for _ in range(copies):
        packed, qs = F.quantize_4bit(W.clone(), blocksize=64, quant_type="nf4", compress_statistics=True)
        packs.append((packed, qs))
    absmax = resolve_absmax(packs[0][1])
    W_ref = dequant_nf4_reference(packs[0][0], absmax, N, K)  # fp32 oracle of the quantised weight
    return packs, W_ref, W_ref.to(torch.bfloat16)


def run(out_path: str, with_triton: bool, force: bool) -> None:
    assert_gpu_free(force)
    import torch
    from bitsandbytes import matmul_4bit
    import bitsandbytes.functional as F

    from aha.kernels.roofline import RTX3060, achieved, gemm_flops, nf4_weight_bytes, utilisation

    if with_triton:
        from aha.kernels.nf4_gemm import absmax_fp16_for, nf4_gemm

    torch.manual_seed(0)
    rows = []
    for name, N, K in QWEN_SHAPES:
        copies = max(1, math.ceil(ROTATE_BYTES / (N * K // 2)))
        packs, W_ref, W16s = make_weights(N, K, copies)
        W16 = [W16s.clone() for _ in range(copies)]
        for M in MS:
            x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
            ref = x.float() @ W_ref.t()
            nbytes = nf4_weight_bytes(N, K) + x.numel() * 2 + M * N * 2
            flops = gemm_flops(M, N, K)
            cands = {
                "bnb fused": lambda i: matmul_4bit(x, packs[i % copies][0].t(), packs[i % copies][1]),
                "bnb dequant+cuBLAS": lambda i: torch.nn.functional.linear(
                    x, F.dequantize_4bit(packs[i % copies][0], packs[i % copies][1])),
                "cuBLAS bf16 weight": lambda i: torch.nn.functional.linear(x, W16[i % copies]),
            }
            if with_triton:
                am16 = [absmax_fp16_for(qs) for _, qs in packs]
                cands["triton nf4"] = lambda i: nf4_gemm(x, packs[i % copies][0], am16[i % copies], N, K)
            for cname, fn in cands.items():
                with torch.no_grad():
                    out = fn(0)
                    ms = time_ms(fn)
                b = nbytes if "bf16" not in cname else N * K * 2 + x.numel() * 2 + M * N * 2
                ach = achieved(b, flops, ms / 1e3)
                util = utilisation(ach)
                rows.append({"shape": f"{name} {N}x{K}", "M": M, "kernel": cname, "ms": ms,
                             "GB/s": ach["gbs"], "bw%": util["bandwidth"], "TFLOPS": ach["tflops"],
                             "tc%": util["compute"], "rel_err": rel_fro_error(out, ref)})
                print(rows[-1])
    cols = ["shape", "M", "kernel", "ms", "GB/s", "bw%", "TFLOPS", "tc%", "rel_err"]
    header = (f"# 4-bit GEMM, decode shapes ({RTX3060['name']})\n\n"
              "bytes = packed nibbles + fp16 absmax + activations; bw% against 360 GB/s; tc% against 25.6 TFLOPS "
              "(fp32 accumulate). rel_err = ||out - ref||_F / ||ref||_F against an fp32 product of the dequantised weight.")
    write_report(out_path, [render_table(rows, cols, "all shapes")], header)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/kernels/gemm_4bit.md")
    p.add_argument("--triton", action="store_true", help="include aha.kernels.nf4_gemm (Task 7)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(a.out, a.triton, a.force)
