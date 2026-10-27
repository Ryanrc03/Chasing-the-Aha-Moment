"""Kernel-level attribution of GPU time: is a step bound by moving parameters
(bandwidth) or by compute (operators), and which operators dominate?

Pure helpers here (categorisation, trace aggregation, roofline arithmetic) are
unit-testable; the GPU work lives in ``scripts/profile_step.py``.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

# RTX 3060 12 GB (GA106): 360 GB/s, ~12.7 TFLOPS FP32 CUDA cores. GeForce Ampere runs the tensor
# cores at half rate with fp32 accumulate, which every bf16 GEMM needs: 25.6 TFLOPS, not the
# 51 TFLOPS (fp16 accumulate) on the spec sheet. See aha.kernels.roofline.
RTX3060 = {"bandwidth_gbs": 360.0, "bf16_tflops": 25.6, "fp32_tflops": 12.7}

CATEGORIES = [
    ("dequant (bnb 4-bit)", re.compile(r"dequant|kDequantize|gemm_4bit|kgemm_4bit|bitsandbytes|nf4", re.I)),
    ("attention", re.compile(r"flash|fmha|attention|attn|xformers|sdpa|_scaled_dot", re.I)),
    ("gemm (matmul)", re.compile(r"gemm|cutlass|cublas|matmul|mm_|bmm|Sgemm|Hgemm|ampere_", re.I)),
    ("cross-entropy / softmax", re.compile(r"cross_entropy|softmax|logsumexp|nll", re.I)),
    ("optimizer", re.compile(r"adam|optimizer|paged|_foreach", re.I)),
    ("memcpy / memset", re.compile(r"Memcpy|Memset|copy_|CopyDevice|cudaMemcpy", re.I)),
    ("norm / rope / activation", re.compile(r"rms|norm|rope|rotary|silu|gelu|swiglu|embedding", re.I)),
    ("elementwise / misc", re.compile(r"elementwise|vectorized|triton_|reduce|fill|cat|index|gather|scatter|where|mul|add", re.I)),
]


def categorize(kernel_name: str) -> str:
    for label, pattern in CATEGORIES:
        if pattern.search(kernel_name):
            return label
    return "other"


def aggregate(kernels: list[tuple[str, float]]) -> dict:
    """``[(name, cuda_time_us), ...]`` -> per-category and per-kernel totals."""
    by_cat: dict[str, float] = defaultdict(float)
    by_name: dict[str, float] = defaultdict(float)
    for name, us in kernels:
        by_cat[categorize(name)] += us
        by_name[name] += us
    total = sum(by_cat.values()) or 1.0
    cats = sorted(by_cat.items(), key=lambda kv: -kv[1])
    names = sorted(by_name.items(), key=lambda kv: -kv[1])
    return {
        "total_us": total,
        "categories": [{"category": c, "us": us, "share": us / total} for c, us in cats],
        "top_kernels": [{"kernel": n[:110], "us": us, "share": us / total} for n, us in names[:25]],
    }


def kernels_from_chrome_trace(path: Path) -> list[tuple[str, float]]:
    """Extract (name, dur_us) for GPU kernel events from a PyTorch/vLLM chrome trace."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    out = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", "")).lower()
        if cat in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime_kernel") or cat.startswith("kernel"):
            out.append((e.get("name", "?"), float(e.get("dur", 0.0))))
    return out


def gpu_busy_fraction(kernels: list[tuple[str, float]], wall_us: float) -> float:
    """Sum of kernel time / wall time (ignores kernel overlap, so it is an upper bound)."""
    return min(1.0, sum(us for _, us in kernels) / wall_us) if wall_us > 0 else 0.0


def decode_roofline(weight_bytes: float, tokens_per_s: float, batch: int,
                    gpu: dict = RTX3060) -> dict:
    """Bandwidth floor for decode: every generated step must stream all weights once.

    With a batch of ``batch`` sequences one weight pass yields ``batch`` tokens.
    """
    step_floor_ms = weight_bytes / (gpu["bandwidth_gbs"] * 1e9) * 1e3
    achieved_step_ms = batch / tokens_per_s * 1e3
    return {
        "weight_bytes": weight_bytes,
        "bandwidth_floor_ms_per_step": step_floor_ms,
        "achieved_ms_per_step": achieved_step_ms,
        "bandwidth_utilisation": step_floor_ms / achieved_step_ms if achieved_step_ms else 0.0,
    }


def train_roofline(params: float, tokens: int, step_s: float, gpu: dict = RTX3060) -> dict:
    """Compute floor for forward+backward (~6 FLOPs per parameter-token, +recompute ~8)."""
    flops = 8.0 * params * tokens
    return {
        "flops": flops,
        "achieved_tflops": flops / step_s / 1e12,
        "compute_utilisation_bf16": flops / step_s / (gpu["bf16_tflops"] * 1e12),
    }


def render(title: str, agg: dict, extra_lines: list[str] = ()) -> str:
    lines = [f"## {title}\n", f"GPU kernel time accounted: {agg['total_us'] / 1e3:.1f} ms\n",
             "| category | ms | share |", "|---|---|---|"]
    for c in agg["categories"]:
        lines.append(f"| {c['category']} | {c['us'] / 1e3:.1f} | {c['share']:.1%} |")
    lines += ["", "| top kernels | ms | share |", "|---|---|---|"]
    for k in agg["top_kernels"][:15]:
        lines.append(f"| `{k['kernel']}` | {k['us'] / 1e3:.1f} | {k['share']:.1%} |")
    lines.append("")
    lines += list(extra_lines)
    lines.append("")
    return "\n".join(lines)
