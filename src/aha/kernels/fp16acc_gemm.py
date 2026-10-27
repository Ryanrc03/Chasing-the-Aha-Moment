"""GEMM with fp16 accumulators, promoted to fp32 every ``promote_every`` K-blocks.

GeForce Ampere issues fp16 tensor-core MMAs at twice the rate when the accumulator is fp16.
cuBLAS (through PyTorch) always accumulates in fp32 (bench/bench_cublas.py: fp16 is no faster
than bf16), so the only way to measure the gain is a kernel of our own. Numerics: an fp16 partial
sum over PROMOTE_EVERY*BLOCK_K products of O(0.5 * 0.02) magnitude stays far from overflow;
precision loss is bounded by the chunk length. Benchmark-only: nothing in training calls it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_chunked_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                         stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         PROMOTE_EVERY: tl.constexpr, ACC_FP16: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    CHUNK: tl.constexpr = PROMOTE_EVERY * BLOCK_K
    for kc in range(0, K, CHUNK):
        if ACC_FP16:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, CHUNK, BLOCK_K):
            k = kc + kk
            a = tl.load(a_ptrs + k * stride_ak, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
            b = tl.load(b_ptrs + k * stride_bk, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
            if ACC_FP16:
                acc = tl.dot(a, b, acc, out_dtype=tl.float16)
            else:
                acc = tl.dot(a, b, acc)
        acc32 += acc.to(tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc32.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm_fp16acc(a: torch.Tensor, b: torch.Tensor, promote_every: int = 4, acc_fp16: bool = True,
                 block_m: int = 128, block_n: int = 128, block_k: int = 32, num_warps: int = 4,
                 num_stages: int = 3) -> torch.Tensor:
    """a [M, K] fp16 @ b [K, N] fp16 -> [M, N] fp16. ``acc_fp16=False`` gives the same kernel with fp32
    accumulation, to separate Triton's codegen gap from the accumulate-rate gain."""
    assert a.dtype == torch.float16 and b.dtype == torch.float16 and a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    chunk = promote_every * block_k
    if K % chunk != 0:
        raise ValueError(f"K={K} must be a multiple of promote_every*block_k={chunk}")
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _gemm_chunked_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                               c.stride(0), c.stride(1),
                               BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, PROMOTE_EVERY=promote_every,
                               ACC_FP16=acc_fp16, num_warps=num_warps, num_stages=num_stages)
    return c
