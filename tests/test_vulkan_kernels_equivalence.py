"""The Vulkan kernels must compute what PyTorch computes.

These are hand-written GLSL behind a pybind11 module, with no test that
compared their output to the PyTorch path they replace. The DirectML patches
had drifted into a TypeError and nothing noticed for the same reason: the
code that was supposed to check it did not exist. A kernel that runs and
returns fp16 of the right shape is not evidence of anything - only a
numerical comparison is.

fp16 accumulation puts a floor under the achievable agreement, so the
tolerances are ~1e-2 relative rather than exact.
"""
import pytest
import torch

vkl = pytest.importorskip("usaf.qwen3_layer_vk")

pytestmark = pytest.mark.skipif(not getattr(vkl, "HAS_VK", False),
                                reason="Vulkan module not built")


def _rel(got, want):
    return float(
        (got.float() - want.float()).abs().max()
        / want.float().abs().max().clamp_min(1e-6)
    )


@pytest.mark.parametrize("m,k", [(128, 96), (64, 64), (256, 128)])
def test_gemm_vk_matches_matmul(m, k):
    torch.manual_seed(0)
    a = torch.randn(8, m, dtype=torch.float16)
    b = torch.randn(m, k, dtype=torch.float16)
    assert vkl.gemm_vk(a, b).shape == (8, k)
    assert _rel(vkl.gemm_vk(a, b), a @ b) < 1e-2


@pytest.mark.parametrize("hidden", [64, 128, 256])
def test_rmsnorm_vk_matches_torch(hidden):
    torch.manual_seed(0)
    x = torch.randn(8, hidden, dtype=torch.float16)
    w = torch.randn(hidden, dtype=torch.float16)
    want = torch.nn.functional.rms_norm(x, (hidden,), w, 1e-6)
    assert _rel(vkl.rmsnorm_vk(x, w, 1e-6), want) < 1e-2


def _rope_tables(seq, head_dim):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(seq).float(), inv)
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1).half()
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1).half()
    return cos, sin


def _rope_ref(x, cos, sin):
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return (
        xf * cos.unsqueeze(0).unsqueeze(0)
        + rot * sin.unsqueeze(0).unsqueeze(0)
    ).to(x.dtype)


def test_rope_vk_matches_the_hf_convention():
    """cos/sin are full head_dim wide, as transformers passes them. The kernel
    and its PyTorch fallback both scale the full width, so a half-width table
    is a caller error, not something either of them should absorb.
    """
    torch.manual_seed(0)
    b, nh, nkv, seq, hd = 2, 4, 2, 8, 32
    q = torch.randn(b, nh, seq, hd, dtype=torch.float16)
    k = torch.randn(b, nkv, seq, hd, dtype=torch.float16)
    cos, sin = _rope_tables(seq, hd)

    gq, gk = vkl.rope_vk(q, k, cos, sin)
    assert gq.shape == q.shape and gk.shape == k.shape
    assert _rel(gq, _rope_ref(q, cos, sin)) < 1e-2
    assert _rel(gk, _rope_ref(k, cos, sin)) < 1e-2


def test_rope_pytorch_fallback_is_exact():
    """The no-Vulkan path is what runs on a machine without the module built,
    so it is a real code path rather than a stub - and it must agree exactly
    with the reference, not merely approximately.
    """
    torch.manual_seed(0)
    b, nh, nkv, seq, hd = 2, 4, 2, 8, 32
    q = torch.randn(b, nh, seq, hd, dtype=torch.float16)
    k = torch.randn(b, nkv, seq, hd, dtype=torch.float16)
    cos, sin = _rope_tables(seq, hd)

    was = vkl.HAS_VK
    vkl.HAS_VK = False
    try:
        fq, fk = vkl.rope_vk(q, k, cos, sin)
    finally:
        vkl.HAS_VK = was

    assert torch.equal(fq, _rope_ref(q, cos, sin))
    assert torch.equal(fk, _rope_ref(k, cos, sin))
