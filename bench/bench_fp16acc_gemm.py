#!/usr/bin/env python
"""Is fp16 accumulation worth it on a GeForce card? Triton GEMM with fp16 vs fp32 accumulators vs
cuBLAS, at training shapes, with error against an fp32 product.

    python bench/bench_fp16acc_gemm.py --out docs/kernels/fp16acc_gemm.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.harness import assert_gpu_free, rel_fro_error, render_table, time_ms, write_report  # noqa: E402

M = 2400
SHAPES = [("q/o_proj", 2048, 2048), ("gate_up", 2048, 11008), ("down", 11008, 2048)]  # (name, K, N)


def run(out_path: str, force: bool) -> None:
    assert_gpu_free(force)
    import torch

    from aha.kernels.fp16acc_gemm import gemm_fp16acc
    from aha.kernels.roofline import achieved, gemm_flops, utilisation

    torch.manual_seed(0)
    rows = []
    for name, K, N in SHAPES:
        a32 = torch.randn(M, K, device="cuda") * 0.5
        b32 = torch.randn(K, N, device="cuda") * 0.02
        ref = a32 @ b32
        a, b = a32.half(), b32.half()
        flops = gemm_flops(M, N, K)
        cands = {
            "cuBLAS fp16 (fp32 acc)": lambda i: a @ b,
            "triton fp32 acc": lambda i: gemm_fp16acc(a, b, acc_fp16=False),
            "triton fp16 acc, promote 1": lambda i: gemm_fp16acc(a, b, promote_every=1),
            "triton fp16 acc, promote 4": lambda i: gemm_fp16acc(a, b, promote_every=4),
            "triton fp16 acc, promote 8": lambda i: gemm_fp16acc(a, b, promote_every=8),
        }
        for cname, fn in cands.items():
            try:
                out = fn(0)
                ms = time_ms(fn, iters=30, warmup=5)
            except Exception as exc:  # noqa: BLE001 - record compile failures as a result
                rows.append({"shape": f"{name} K{K} N{N}", "kernel": cname, "ms": float("nan"), "TFLOPS": float("nan"),
                             "fp32acc%": float("nan"), "fp16acc%": float("nan"), "rel_err": float("nan"), "cos": float("nan"),
                             "note": str(exc)[:80]})
                print(rows[-1])
                continue
            ach = achieved((M * K + K * N + M * N) * 2, flops, ms / 1e3)
            rows.append({"shape": f"{name} K{K} N{N}", "kernel": cname, "ms": ms, "TFLOPS": ach["tflops"],
                         "fp32acc%": utilisation(ach)["compute"], "fp16acc%": utilisation(ach, accumulate="fp16")["compute"],
                         "rel_err": rel_fro_error(out, ref),
                         "cos": torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item(),
                         "note": ""})
            print(rows[-1])
    cols = ["shape", "kernel", "ms", "TFLOPS", "fp32acc%", "fp16acc%", "rel_err", "cos", "note"]
    header = ("# fp16-accumulate GEMM feasibility (RTX 3060)\n\n"
              "fp32acc% = share of the 25.6 TFLOPS fp32-accumulate peak; fp16acc% = share of the 51.2 TFLOPS fp16-accumulate peak. "
              "cos = cosine similarity to the fp32 product (stands in for gradient direction agreement).")
    write_report(out_path, [render_table(rows, cols, "all shapes")], header)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/kernels/fp16acc_gemm.md")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(a.out, a.force)
