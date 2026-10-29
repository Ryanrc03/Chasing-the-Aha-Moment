import pytest

torch = pytest.importorskip("torch", reason="optional GPU tests require PyTorch")
pytest.importorskip("triton", reason="optional GPU tests require Triton")
pytest.importorskip("bitsandbytes", reason="optional GPU tests require bitsandbytes")

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


def test_patch_routes_small_m_and_delegates_large_m():
    import bitsandbytes

    from aha.kernels import nf4_gemm as mod

    x, packed, qs, ref = _case(2048, 2048, 8)
    with torch.no_grad():
        before = bitsandbytes.matmul_4bit(x, packed.t(), qs)
    mod.patch_bitsandbytes()
    try:
        assert bitsandbytes.matmul_4bit is mod._patched_matmul_4bit
        with torch.no_grad():
            after = bitsandbytes.matmul_4bit(x, packed.t(), qs)
            assert _rel(after, ref) <= 1.5 * _rel(before, ref) + 1e-4
            # 3-D activations keep their shape
            after3 = bitsandbytes.matmul_4bit(x.unsqueeze(0), packed.t(), qs)
            assert after3.shape == (1, 8, 2048)
            # large M is delegated to the original implementation
            xb = torch.randn(300, 2048, device="cuda").to(torch.bfloat16)
            big = bitsandbytes.matmul_4bit(xb, packed.t(), qs)
            assert torch.equal(big, mod._ORIGINAL_MATMUL_4BIT(xb, packed.t(), qs))
    finally:
        mod.unpatch_bitsandbytes()
    assert bitsandbytes.matmul_4bit is mod._ORIGINAL_MATMUL_4BIT


@pytest.mark.parametrize("promote_every", [1, 4])
def test_fp16acc_gemm_close_to_fp32(promote_every):
    from aha.kernels.fp16acc_gemm import gemm_fp16acc

    torch.manual_seed(0)
    a = (torch.randn(256, 2048, device="cuda") * 0.5).half()
    b = (torch.randn(2048, 512, device="cuda") * 0.02).half()
    ref = a.float() @ b.float()
    out = gemm_fp16acc(a, b, promote_every=promote_every)
    assert out.dtype == torch.float16
    assert _rel(out, ref) < 5e-3
