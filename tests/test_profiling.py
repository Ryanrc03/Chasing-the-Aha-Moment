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


def test_attribute_kernels_maps_fill_to_python_frame(tmp_path):
    from aha.profiling import attribute_kernels, render_attribution

    trace = {"traceEvents": [
        # python frames (with_stack=True): outer user frame, inner torch frame
        {"ph": "X", "cat": "python_function", "tid": 1, "ts": 0, "dur": 100,
         "name": "site-packages/peft/tuners/lora/layer.py(400): forward"},
        {"ph": "X", "cat": "python_function", "tid": 1, "ts": 10, "dur": 50,
         "name": "site-packages/torch/nn/functional.py(10): linear"},
        {"ph": "X", "cat": "cpu_op", "tid": 1, "ts": 20, "dur": 5, "name": "aten::fill_", "args": {"External id": 7}},
        {"ph": "X", "cat": "cpu_op", "tid": 1, "ts": 60, "dur": 5, "name": "aten::copy_", "args": {"External id": 8}},
        {"ph": "X", "cat": "kernel", "ts": 1000, "dur": 30.0, "name": "void at::native::vectorized_elementwise_kernel<FillFunctor>", "args": {"External id": 7}},
        {"ph": "X", "cat": "kernel", "ts": 1100, "dur": 10.0, "name": "void at::native::vectorized_elementwise_kernel<bfloat16_copy_kernel>", "args": {"External id": 8}},
        {"ph": "X", "cat": "kernel", "ts": 1200, "dur": 500.0, "name": "ampere_bf16_gemm", "args": {"External id": 9}},
    ]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(trace))
    rows = attribute_kernels(p)
    assert rows[0]["op"] == "aten::fill_" and rows[0]["us"] == 30.0
    assert rows[0]["frame"] == "site-packages/peft/tuners/lora/layer.py(400): forward"
    assert rows[1]["op"] == "aten::copy_" and rows[1]["frame"].endswith("layer.py(400): forward")
    md = render_attribution(rows, total_us=540.0)
    assert "aten::fill_" in md and "5.6%" in md
