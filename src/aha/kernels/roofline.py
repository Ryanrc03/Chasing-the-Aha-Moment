"""Hardware ceilings for the RTX 3060 (GA106) and the arithmetic to compare a kernel against them.

GeForce Ampere halves the tensor-core rate when accumulating in fp32, which is what every
bf16 GEMM must do; the 51 TFLOPS figure on spec sheets is fp16-accumulate only.
"""

from __future__ import annotations

RTX3060 = {
    "name": "RTX 3060 12 GB (GA106)",
    "bandwidth_gbs": 360.0,
    "tflops_fp32_acc": 25.6,   # fp16/bf16 tensor core, fp32 accumulate (cuBLAS bf16 path)
    "tflops_fp16_acc": 51.2,   # fp16 tensor core, fp16 accumulate
    "tflops_fp32_cuda": 12.7,  # CUDA cores
    "l2_bytes": 3 * 2**20,
}


def gemm_flops(M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K


def nf4_weight_bytes(N: int, K: int, blocksize: int = 64, absmax_bytes: int = 2) -> int:
    """Bytes streamed per use of an NF4 weight: packed nibbles plus one absmax per block."""
    return N * K // 2 + (N * K // blocksize) * absmax_bytes


def achieved(bytes_moved: float, flops: float, seconds: float) -> dict:
    return {"gbs": bytes_moved / 1e9 / seconds, "tflops": flops / 1e12 / seconds}


def utilisation(ach: dict, gpu: dict = RTX3060, accumulate: str = "fp32") -> dict:
    peak = gpu["tflops_fp32_acc"] if accumulate == "fp32" else gpu["tflops_fp16_acc"]
    return {"bandwidth": ach["gbs"] / gpu["bandwidth_gbs"], "compute": ach["tflops"] / peak}
