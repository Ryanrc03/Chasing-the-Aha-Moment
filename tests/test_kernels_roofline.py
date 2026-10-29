import pytest

from aha.kernels.roofline import RTX3060, achieved, gemm_flops, nf4_weight_bytes, utilisation


def test_constants():
    assert RTX3060["bandwidth_gbs"] == 360.0
    assert RTX3060["tflops_fp32_acc"] == 25.6
    assert RTX3060["tflops_fp16_acc"] == 51.2


def test_bytes_and_flops():
    assert gemm_flops(6, 2048, 2048) == 2 * 6 * 2048 * 2048
    # packed nibbles + one fp16 absmax per 64 weights
    assert nf4_weight_bytes(2048, 2048) == 2048 * 2048 // 2 + 2048 * 2048 // 64 * 2
    assert nf4_weight_bytes(2048, 2048, absmax_bytes=4) == 2048 * 2048 // 2 + 2048 * 2048 // 64 * 4


def test_achieved_and_utilisation():
    a = achieved(bytes_moved=360e9, flops=25.6e12, seconds=1.0)
    assert a["gbs"] == pytest.approx(360.0)
    assert a["tflops"] == pytest.approx(25.6)
    u = utilisation(a)
    assert u["bandwidth"] == pytest.approx(1.0)
    assert u["compute"] == pytest.approx(1.0)
    assert utilisation(a, accumulate="fp16")["compute"] == pytest.approx(0.5)
