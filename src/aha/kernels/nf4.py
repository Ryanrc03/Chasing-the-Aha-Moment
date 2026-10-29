"""NF4 (bitsandbytes) layout in plain torch: the oracle every kernel in this package is tested against.

Layout facts (bitsandbytes 0.50): a [N, K] weight is flattened row-major, quantised in blocks of
``blocksize`` (64) values to 4-bit code indices, and packed two per byte with the first value in the
high nibble -> uint8 tensor of shape [(N*K)//2, 1]. One fp32 absmax per block; with
``compress_statistics=True`` the absmax tensor is itself 8-bit block-quantised (blocksize 256,
``quant_state.state2``) with a scalar ``offset`` added back after dequantisation.
"""

from __future__ import annotations

import torch

# bitsandbytes' NF4 codebook (create_normal_map(offset=0.9677083)), ascending.
NF4_CODE = torch.tensor(
    [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
    ],
    dtype=torch.float32,
)


def pack_nf4_reference(idx: torch.Tensor) -> torch.Tensor:
    """4-bit code indices (uint8, even length) -> packed [(n)//2, 1] uint8, first index in the high nibble."""
    idx = idx.reshape(-1)
    assert idx.numel() % 2 == 0
    return ((idx[0::2] << 4) | idx[1::2]).to(torch.uint8).reshape(-1, 1)


def unpack_nf4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Packed uint8 -> code indices (uint8) of length ``numel``."""
    b = packed.reshape(-1)
    idx = torch.stack([b >> 4, b & 0xF], dim=-1).reshape(-1)
    return idx[:numel]


def resolve_absmax(quant_state) -> torch.Tensor:
    """fp32 absmax per block, undoing bitsandbytes' nested 8-bit statistics when present."""
    import bitsandbytes.functional as F

    if quant_state.nested:
        absmax = F.dequantize_blockwise(quant_state.absmax, quant_state.state2) + quant_state.offset
    else:
        absmax = quant_state.absmax
    return absmax.float()


def dequant_nf4_reference(packed: torch.Tensor, absmax: torch.Tensor, N: int, K: int,
                          blocksize: int = 64) -> torch.Tensor:
    """fp32 [N, K] weight from packed nibbles and per-block absmax. Slow, obviously correct."""
    idx = unpack_nf4(packed, N * K).long()
    code = NF4_CODE.to(idx.device)
    vals = code[idx].reshape(-1, blocksize) * absmax.float().reshape(-1, 1)
    return vals.reshape(N, K)
