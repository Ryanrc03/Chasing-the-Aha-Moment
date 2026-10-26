import json

import pytest

from aha.profiling import (
    aggregate,
    categorize,
    decode_roofline,
    gpu_busy_fraction,
    kernels_from_chrome_trace,
    render,
    train_roofline,
)


@pytest.mark.parametrize("name, cat", [
    ("void kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, 2>(...)", "dequant (bnb 4-bit)"),
    ("kgemm_4bit_inference_naive<__nv_bfloat16, 128, 16>", "dequant (bnb 4-bit)"),
    ("ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn", "gemm (matmul)"),
    ("flash_fwd_kernel<...>", "attention"),
    ("void at::native::vectorized_elementwise_kernel<4, ...>", "elementwise / misc"),
    ("Memcpy DtoD (Device -> Device)", "memcpy / memset"),
    ("triton_red_fused__log_softmax_0", "cross-entropy / softmax"),
    ("void rms_norm_kernel<...>", "norm / rope / activation"),
    ("totally_unknown_kernel", "other"),
])
def test_categorize(name, cat):
    assert categorize(name) == cat


def test_aggregate_shares():
    agg = aggregate([("ampere_gemm", 600.0), ("kDequantizeBlockwise", 300.0), ("Memcpy DtoD", 100.0)])
    assert agg["total_us"] == 1000.0
    assert agg["categories"][0] == {"category": "gemm (matmul)", "us": 600.0, "share": 0.6}
    assert agg["top_kernels"][1]["kernel"] == "kDequantizeBlockwise"


def test_chrome_trace_parsing(tmp_path):
    trace = {"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "ampere_gemm", "dur": 10.0},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "dur": 50.0},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "dur": 2.5},
        {"ph": "M", "name": "process_name"},
    ]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(trace))
    assert kernels_from_chrome_trace(p) == [("ampere_gemm", 10.0), ("Memcpy HtoD", 2.5)]


def test_rooflines():
    assert gpu_busy_fraction([("k", 500.0)], 1000.0) == 0.5
    r = decode_roofline(weight_bytes=2.35 * 2**30, tokens_per_s=174.0, batch=6)
    assert 6.5 < r["bandwidth_floor_ms_per_step"] < 7.5
    assert 34 < r["achieved_ms_per_step"] < 35
    assert 0.15 < r["bandwidth_utilisation"] < 0.25
    t = train_roofline(params=3.1e9, tokens=6 * 600, step_s=2.0)
    assert t["achieved_tflops"] == pytest.approx(8 * 3.1e9 * 3600 / 2.0 / 1e12)


def test_render():
    md = render("T", aggregate([("ampere_gemm", 1000.0)]), ["- note"])
    assert "## T" in md and "gemm (matmul)" in md and "- note" in md
