"""Small-M NF4 weight-only GEMM in Triton: y[M, N] = x[M, K] @ W[N, K]^T with W packed as bnb NF4.

Why: at decode (M = 6..32) bitsandbytes' fused kernel streams the weight at 12-22% of the 3060's
bandwidth (bench/bench_gemm_4bit.py). This kernel keeps the weight packed until it is in
registers: each program owns a BLOCK_N row stripe and a K slice (split-K, fp32 atomics), walks K
in blocks of 64 (= the NF4 blocksize, so one absmax per row per block), loads the packed bytes,
turns each byte into its two bf16 codebook values with a PTX ``prmt`` byte-permute tree (7 prmt
per byte, codebook held in registers; no memory lookup), feeds two tensor-core dots (even k from
the high nibbles, odd k from the low ones) and scales the per-block partial product by the
absmax afterwards. fp32 accumulate, bf16 out.

Design notes from the probes (scratchpad nf4_experiment*.py, 2026-10-27):
- a 16-entry codebook gather from global/L1 is the single most expensive thing Triton can emit
  here (two fp32 gathers: 0.37 ms on gate_up; one packed int32 gather: 0.22 ms; prmt: 0.14 ms);
- a 15-way tl.where select tree costs about the same as prmt;
- with the lookup removed entirely the kernel reaches ~143 GB/s (40%), which is this design's
  ceiling in Triton; going further needs a CUDA kernel with a shared-memory LUT and cp.async.

Nested absmax statistics are resolved once per weight into an fp16 vector (``absmax_fp16_for``)
and cached by the identity of the quant state's absmax tensor; ~95 MB for the whole 3B model.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from aha.kernels.nf4 import NF4_CODE, resolve_absmax

MAX_M = 32
BLOCKSIZE = 64

# The codebook as bf16 bit patterns, two entries per 32-bit register (entry 2i in the low half).
_BITS = [int(v) & 0xFFFF for v in NF4_CODE.to(torch.bfloat16).view(torch.int16).tolist()]
_C32 = [(_BITS[2 * i] | (_BITS[2 * i + 1] << 16)) & 0xFFFFFFFF for i in range(8)]

# $0 <- packed (bf16(code[lo]) << 16) | bf16(code[hi]) for the byte $1 = (hi << 4) | lo.
# Level 1: four prmt pick entry (q & 3) from each 4-entry pool for both nibbles at once;
# level 2 chooses the pool pair by bit 2, level 3 by bit 3.
_PRMT_ASM = """
{
    .reg .b32 r0, r1, r2, r3, s0, s1, sa, sb, sc;
    .reg .b32 hi, lo, bh, bl;
    .reg .b32 c0, c1, c2, c3, c4, c5, c6, c7;
    mov.b32 c0, %C0%; mov.b32 c1, %C1%; mov.b32 c2, %C2%; mov.b32 c3, %C3%;
    mov.b32 c4, %C4%; mov.b32 c5, %C5%; mov.b32 c6, %C6%; mov.b32 c7, %C7%;
    shr.u32 hi, $1, 4;
    and.b32 lo, $1, 15;
    and.b32 bh, hi, 3;  shl.b32 bh, bh, 1;
    and.b32 bl, lo, 3;  shl.b32 bl, bl, 1;
    add.u32 sa, bh, 1;  shl.b32 sa, sa, 4;  or.b32 sa, sa, bh;
    add.u32 sb, bl, 1;  shl.b32 sb, sb, 12; or.b32 sa, sa, sb;
    shl.b32 sb, bl, 8;  or.b32 sa, sa, sb;
    prmt.b32 r0, c0, c1, sa;
    prmt.b32 r1, c2, c3, sa;
    prmt.b32 r2, c4, c5, sa;
    prmt.b32 r3, c6, c7, sa;
    and.b32 bh, hi, 4;  shr.u32 bh, bh, 2;  shl.b32 bh, bh, 2;
    and.b32 bl, lo, 4;  shr.u32 bl, bl, 2;  shl.b32 bl, bl, 2;
    add.u32 sc, bh, 1;  shl.b32 sc, sc, 4;  or.b32 sc, sc, bh;
    add.u32 sb, bl, 2;  add.u32 sa, sb, 1;  shl.b32 sa, sa, 12; or.b32 sc, sc, sa;
    shl.b32 sb, sb, 8;  or.b32 sc, sc, sb;
    prmt.b32 s0, r0, r1, sc;
    prmt.b32 s1, r2, r3, sc;
    and.b32 bh, hi, 8;  shr.u32 bh, bh, 1;
    and.b32 bl, lo, 8;  shr.u32 bl, bl, 1;
    add.u32 sc, bh, 1;  shl.b32 sc, sc, 4;  or.b32 sc, sc, bh;
    add.u32 sb, bl, 2;  add.u32 sa, sb, 1;  shl.b32 sa, sa, 12; or.b32 sc, sc, sa;
    shl.b32 sb, sb, 8;  or.b32 sc, sc, sb;
    prmt.b32 $0, s0, s1, sc;
}
"""
for _i, _c in enumerate(_C32):
    _PRMT_ASM = _PRMT_ASM.replace(f"%C{_i}%", "0x%08x" % _c)
PRMT_ASM = tl.constexpr(_PRMT_ASM)


@triton.jit
def _nf4_gemm_kernel(a_ptr, w_ptr, absmax_ptr, c_ptr,
                     M, N, K, stride_am, stride_cm,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    HALF: tl.constexpr = BLOCK_K // 2
    offs_h = tl.arange(0, HALF)                      # byte index inside the K block
    kblocks = K // BLOCK_K
    kb_per = kblocks // SPLIT_K
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kb in range(pid_k * kb_per, (pid_k + 1) * kb_per):
        k0 = kb * BLOCK_K
        a_even = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + 2 * offs_h)[None, :],
                         mask=mask_m[:, None], other=0.0)
        a_odd = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + 2 * offs_h + 1)[None, :],
                        mask=mask_m[:, None], other=0.0)
        w_bytes = tl.load(w_ptr + (offs_n[:, None] * K + k0) // 2 + offs_h[None, :],
                          mask=mask_n[:, None], other=0)              # [BLOCK_N, HALF] uint8
        scale = tl.load(absmax_ptr + offs_n * kblocks + kb, mask=mask_n, other=0.0).to(tl.float32)
        v = tl.inline_asm_elementwise(PRMT_ASM, "=r,r", [w_bytes.to(tl.int32)], dtype=tl.int32,
                                      is_pure=True, pack=1)
        w_hi = (v & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)        # even k
        w_lo = ((v >> 16) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)  # odd k
        part = tl.dot(w_hi, tl.trans(a_even))
        part += tl.dot(w_lo, tl.trans(a_odd))
        acc += part * scale[:, None]
    ptrs = c_ptr + offs_m[None, :] * stride_cm + offs_n[:, None]
    msk = mask_m[None, :] & mask_n[:, None]
    if SPLIT_K > 1:
        tl.atomic_add(ptrs, acc, mask=msk)
    else:
        tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=msk)


_ABSMAX_CACHE: dict[int, torch.Tensor] = {}


def absmax_fp16_for(quant_state) -> torch.Tensor:
    """fp16 absmax per 64-block for a bnb quant state, cached on the absmax tensor's identity."""
    key = quant_state.absmax.data_ptr()
    hit = _ABSMAX_CACHE.get(key)
    if hit is None:
        hit = resolve_absmax(quant_state).to(torch.float16).contiguous()
        _ABSMAX_CACHE[key] = hit
    return hit


def _split_k(K: int) -> int:
    kblocks = K // BLOCKSIZE
    for s in (4, 2):
        if kblocks % s == 0:
            return s
    return 1


def nf4_gemm(x: torch.Tensor, packed: torch.Tensor, absmax16: torch.Tensor, N: int, K: int,
             out: torch.Tensor | None = None, block_n: int = 64, num_warps: int = 4,
             num_stages: int = 3) -> torch.Tensor:
    """x [M, K] bf16 (M <= 32), packed bnb NF4 weight of logical shape [N, K] -> [M, N] bf16."""
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError(f"x must be 2-D bf16, got {tuple(x.shape)} {x.dtype}")
    M = x.shape[0]
    if M > MAX_M:
        raise ValueError(f"M={M} > {MAX_M}: this kernel is for decode-sized batches")
    if x.shape[1] != K or K % BLOCKSIZE != 0:
        raise ValueError(f"K must equal x.shape[1] and be a multiple of {BLOCKSIZE}, got K={K}, x={tuple(x.shape)}")
    if packed.numel() != N * K // 2 or absmax16.numel() != N * K // BLOCKSIZE:
        raise ValueError("packed/absmax sizes do not match N x K")
    x = x.contiguous()
    block_m = 16 if M <= 16 else 32
    split_k = _split_k(K)
    grid = (triton.cdiv(N, block_n), split_k)
    if split_k > 1:
        acc = torch.zeros((M, N), dtype=torch.float32, device=x.device)
        _nf4_gemm_kernel[grid](x, packed, absmax16, acc, M, N, K, x.stride(0), acc.stride(0),
                               BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=BLOCKSIZE, SPLIT_K=split_k,
                               num_warps=num_warps, num_stages=num_stages)
        if out is None:
            return acc.to(torch.bfloat16)
        out.copy_(acc)
        return out
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    _nf4_gemm_kernel[grid](x, packed, absmax16, out, M, N, K, x.stride(0), out.stride(0),
                           BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=BLOCKSIZE, SPLIT_K=1,
                           num_warps=num_warps, num_stages=num_stages)
    return out


# --- bitsandbytes integration -------------------------------------------------------------
# vLLM's bitsandbytes linear method does ``from bitsandbytes import matmul_4bit`` inside the call,
# so rebinding the package attribute reaches the engine (which Unsloth runs in this process,
# VLLM_ENABLE_V1_MULTIPROCESSING=0). Apply before the engine is built so the warm-up run fills the
# absmax cache before CUDA-graph capture.

_ORIGINAL_MATMUL_4BIT = None


def _patched_matmul_4bit(A, B, quant_state, out=None, bias=None):
    if (
        quant_state is not None
        and quant_state.quant_type == "nf4"
        and quant_state.blocksize == BLOCKSIZE
        and A.dtype == torch.bfloat16
        and not (torch.is_grad_enabled() and A.requires_grad)
        and A.shape[-1] == quant_state.shape[1]
        and quant_state.shape[1] % BLOCKSIZE == 0
        and 1 <= A.numel() // A.shape[-1] <= MAX_M
    ):
        N, K = quant_state.shape
        x2 = A.reshape(-1, K)
        y = nf4_gemm(x2, B.reshape(-1), absmax_fp16_for(quant_state), N, K)
        if bias is not None:
            y = y + bias
        y = y.reshape(*A.shape[:-1], N)
        if out is not None:
            out.copy_(y)
            return out
        return y
    return _ORIGINAL_MATMUL_4BIT(A, B, quant_state, out=out, bias=bias)


def patch_bitsandbytes() -> None:
    global _ORIGINAL_MATMUL_4BIT
    import bitsandbytes

    if bitsandbytes.matmul_4bit is _patched_matmul_4bit:
        return
    _ORIGINAL_MATMUL_4BIT = bitsandbytes.matmul_4bit
    bitsandbytes.matmul_4bit = _patched_matmul_4bit


def unpatch_bitsandbytes() -> None:
    import bitsandbytes

    if _ORIGINAL_MATMUL_4BIT is not None:
        bitsandbytes.matmul_4bit = _ORIGINAL_MATMUL_4BIT
