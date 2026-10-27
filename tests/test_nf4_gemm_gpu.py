import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + triton")


def _case(N, K, M, seed=0):
    import bitsandbytes.functional as F

    from aha.kernels.nf4 import dequant_nf4_reference, resolve_absmax

    torch.manual_seed(seed)
    W = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    packed, qs = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=True)
    x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
    ref = x.float() @ dequant_nf4_reference(packed, resolve_absmax(qs), N, K).t()
    return x, packed, qs, ref


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("N,K", [(256, 2048), (2048, 2048), (2048, 11008)])
@pytest.mark.parametrize("M", [1, 6, 24, 32])
def test_nf4_gemm_matches_reference_within_bnb_error(N, K, M):
    from bitsandbytes import matmul_4bit

    from aha.kernels.nf4_gemm import absmax_fp16_for, nf4_gemm

    x, packed, qs, ref = _case(N, K, M)
    with torch.no_grad():
        theirs = matmul_4bit(x, packed.t(), qs)
        ours = nf4_gemm(x, packed, absmax_fp16_for(qs), N, K)
    assert ours.shape == (M, N) and ours.dtype == torch.bfloat16
    err_ours, err_bnb = _rel(ours, ref), _rel(theirs, ref)
    assert err_ours <= 1.5 * err_bnb + 1e-4, (err_ours, err_bnb)
    assert err_ours < 1e-2


def test_nf4_gemm_rejects_unsupported_shapes():
    from aha.kernels.nf4_gemm import nf4_gemm

    x = torch.zeros(4, 100, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        nf4_gemm(x, torch.zeros(50, 1, dtype=torch.uint8, device="cuda"),
                 torch.zeros(2, dtype=torch.float16, device="cuda"), 1, 100)
    x = torch.zeros(64, 2048, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        nf4_gemm(x, torch.zeros(2048 * 2048 // 2, 1, dtype=torch.uint8, device="cuda"),
                 torch.zeros(2048 * 2048 // 64, dtype=torch.float16, device="cuda"), 2048, 2048)
