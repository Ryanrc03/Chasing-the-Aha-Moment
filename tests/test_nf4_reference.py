import pytest
import torch

from aha.kernels.nf4 import NF4_CODE, dequant_nf4_reference, pack_nf4_reference, unpack_nf4

CUDA = torch.cuda.is_available()


def test_codebook_is_sorted_and_symmetric_endpoints():
    assert NF4_CODE.shape == (16,)
    assert torch.all(NF4_CODE[1:] > NF4_CODE[:-1])
    assert NF4_CODE[0] == -1.0 and NF4_CODE[-1] == 1.0 and NF4_CODE[7] == 0.0


def test_pack_unpack_roundtrip():
    idx = torch.randint(0, 16, (2 * 64 * 3,), dtype=torch.uint8)
    packed = pack_nf4_reference(idx)
    assert packed.shape == (idx.numel() // 2, 1) and packed.dtype == torch.uint8
    assert packed[0, 0] == (int(idx[0]) << 4) | int(idx[1])  # first element in the high nibble
    assert torch.equal(unpack_nf4(packed, idx.numel()), idx)


def test_dequant_reference_scales_each_block():
    N, K = 2, 128  # 4 blocks of 64
    idx = torch.full((N * K,), 15, dtype=torch.uint8)  # code 1.0 everywhere
    absmax = torch.tensor([0.5, 1.0, 2.0, 4.0])
    W = dequant_nf4_reference(pack_nf4_reference(idx), absmax, N, K)
    assert W.shape == (N, K) and W.dtype == torch.float32
    assert W[0, 0] == 0.5 and W[0, 64] == 1.0 and W[1, 0] == 2.0 and W[1, 127] == 4.0


@pytest.mark.skipif(not CUDA, reason="needs bitsandbytes CUDA")
def test_reference_matches_bitsandbytes():
    import bitsandbytes.functional as F

    from aha.kernels.nf4 import resolve_absmax

    torch.manual_seed(0)
    N, K = 256, 2048
    W = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    packed, qs = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=True)
    assert qs.nested
    ours = dequant_nf4_reference(packed, resolve_absmax(qs), N, K)
    theirs = F.dequantize_4bit(packed, qs).float()  # bnb returns the quant state's dtype (bf16)
    assert ((ours - theirs).abs() <= theirs.abs() * 4e-3 + 1e-6).all()  # within bf16 rounding
