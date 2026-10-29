#!/usr/bin/env python
"""Training-shaped GEMMs (M = 6 seq x 400 tokens) through cuBLAS, bf16 and fp16, against the
fp32-accumulate tensor-core peak. Establishes what the fp16-accumulate experiment (P3) must beat.

    python bench/bench_cublas.py --out docs/kernels/cublas_bf16.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.harness import assert_gpu_free, rel_fro_error, render_table, time_ms, write_report  # noqa: E402

M = 2400
SHAPES = [("q/o_proj", 2048, 2048), ("kv_proj", 256, 2048), ("gate_up", 11008, 2048), ("down", 2048, 11008),
          ("lora_A", 32, 2048), ("lora_B", 2048, 32), ("lm_head", 151936, 2048)]


def run(out_path: str, force: bool) -> None:
    assert_gpu_free(force)
    import torch

    from aha.kernels.roofline import RTX3060, achieved, gemm_flops, utilisation

    torch.manual_seed(0)
    rows = []
    for name, N, K in SHAPES:
        x32 = torch.randn(M, K, device="cuda") * 0.5
        w32 = torch.randn(N, K, device="cuda") * 0.02
        ref = x32 @ w32.t()
        flops = gemm_flops(M, N, K)
        nbytes = (M * K + N * K + M * N) * 2
        for dt, label in [(torch.bfloat16, "cuBLAS bf16"), (torch.float16, "cuBLAS fp16"),
                          (torch.float32, "cuBLAS fp32 (cuda cores)")]:
            x, w = x32.to(dt), w32.to(dt)
            fn = lambda i: torch.nn.functional.linear(x, w)  # noqa: E731
            out = fn(0)
            ms = time_ms(fn, iters=30, warmup=5)
            ach = achieved(nbytes if dt != torch.float32 else nbytes * 2, flops, ms / 1e3)
            util = utilisation(ach)
            rows.append({"shape": f"{name} {N}x{K}", "kernel": label, "ms": ms, "TFLOPS": ach["tflops"],
                         "tc%": util["compute"], "GB/s": ach["gbs"], "rel_err": rel_fro_error(out, ref)})
            print(rows[-1])
            del out
    cols = ["shape", "kernel", "ms", "TFLOPS", "tc%", "GB/s", "rel_err"]
    header = (f"# cuBLAS GEMM at training shapes, M={M} ({RTX3060['name']})\n\n"
              "tc% against 25.6 TFLOPS (fp32 accumulate); fp32 rows are CUDA-core GEMMs (12.7 TFLOPS peak).")
    write_report(out_path, [render_table(rows, cols, "all shapes")], header)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/kernels/cublas_bf16.md")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(a.out, a.force)
