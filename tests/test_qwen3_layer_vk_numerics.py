"""The Vulkan kernels, against PyTorch, on the real device.

qwen3_layer_forward_vk runs RMSNorm, two GEMMs, RoPE and a gather/scatter MoE
every call, and none of those went through anything but their own arithmetic.
On a small model the shape errors would show, but a kernel that is subtly wrong
produces a plausible number: fp16 hides it until a layer is deep enough for the
error to matter. So the whole layer is compared against an independent PyTorch
written-from-the-specification implementation, on the same weights.

This only runs where there is a Vulkan device. Everything else in the suite is
about bookkeeping; this is about the arithmetic being right.
"""
import pytest
import torch
import torch.nn.functional as functional

from usaf.qwen3_layer_vk import (
    HAS_VK,
    Qwen3LayerWeights,
    qwen3_layer_forward_vk,
)

pytestmark = pytest.mark.skipif(not HAS_VK, reason="no Vulkan device")

HIDDEN, N_HEADS, N_KV, HEAD_DIM = 64, 4, 2, 16
N_EXP, TOP_K, INTER = 4, 2, 32
BATCH, SEQ = 1, 8
EPS = 1e-6


def _rms(x, w, eps=EPS):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * w.float()).to(x.dtype)


def _rope(x, cos, sin):
    # HF Qwen3 rotate_half: split in half, not interleaved pairs.
    xf = x.float()
    a, b = xf.chunk(2, dim=-1)
    rot = torch.cat((-b, a), dim=-1)
    c = cos.float()[None, None]
    s = sin.float()[None, None]
    return (xf * c + rot * s).to(x.dtype)


def reference(hidden, w, cos, sin, mask, gu, dn):
    """Written from the Qwen3 specification, not from the implementation."""
    n_rep = N_HEADS // N_KV
    hs = hidden.reshape(-1, HIDDEN)
    x = _rms(hs, w["input_layernorm.weight"])
    q = _proj(x, w, "q", BATCH, N_HEADS)
    k = _proj(x, w, "k", BATCH, N_KV)
    v = _proj(x, w, "v", BATCH, N_KV)
    q = _rms(q.reshape(-1, HEAD_DIM), w["self_attn.q_norm.weight"])
    q = q.reshape(BATCH, SEQ, N_HEADS, HEAD_DIM).transpose(1, 2)
    k = _rms(k.reshape(-1, HEAD_DIM), w["self_attn.k_norm.weight"])
    k = k.reshape(BATCH, SEQ, N_KV, HEAD_DIM).transpose(1, 2)
    v = v.reshape(BATCH, SEQ, N_KV, HEAD_DIM).transpose(1, 2)
    q = _rope(q, cos, sin)
    k = _rope(k, cos, sin)
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    att = (q.float() @ k.float().transpose(2, 3)) * (HEAD_DIM ** -0.5)
    if mask is not None:
        att = att + mask.float()[:, :, :SEQ, :SEQ]
    att = functional.softmax(att, dim=-1)
    att = (att @ v.float()).transpose(1, 2).reshape(
        BATCH, SEQ, N_HEADS * HEAD_DIM)
    att = att @ w["self_attn.o_proj.weight"].float().T
    h2 = hidden.float() + att.reshape(BATCH, SEQ, HIDDEN)
    x2 = _rms(h2.reshape(-1, HIDDEN), w["post_attention_layernorm.weight"])
    logits = x2.float() @ w["mlp.gate.weight"].float().T
    probs = functional.softmax(logits, dim=-1)
    tv, ti = torch.topk(probs, TOP_K, dim=-1)
    tv = tv / tv.sum(-1, keepdim=True)
    moe = torch.zeros_like(x2.float())
    for ei in range(N_EXP):
        sel = ti == ei
        tok, pos = sel.nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        g, u = (x2[tok].float() @ gu[ei].float().T).chunk(2, dim=-1)
        act = functional.silu(g) * u
        cur = act @ dn[ei].float().T
        moe.index_add_(0, tok, cur * tv[tok, pos].unsqueeze(-1))
    return (h2.reshape(-1, HIDDEN) + moe).reshape(BATCH, SEQ, HIDDEN)


def _proj(x, w, which, batch, n_heads):
    mat = w["self_attn." + which + "_proj.weight"].float().T
    return (x.float() @ mat).reshape(batch, SEQ, n_heads, HEAD_DIM)


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator().manual_seed(7)

    def r(*shape):
        v = torch.randn(*shape, generator=g) * 0.3
        return v.to(torch.float16)

    def u(*shape):
        v = torch.rand(*shape, generator=g) + 0.5
        return v.to(torch.float16)

    return {
        "input_layernorm.weight": torch.ones(HIDDEN, dtype=torch.float16),
        "post_attention_layernorm.weight": u(HIDDEN),
        "self_attn.q_norm.weight": u(HEAD_DIM),
        "self_attn.k_norm.weight": u(HEAD_DIM),
        "self_attn.q_proj.weight": r(N_HEADS * HEAD_DIM, HIDDEN),
        "self_attn.k_proj.weight": r(N_KV * HEAD_DIM, HIDDEN),
        "self_attn.v_proj.weight": r(N_KV * HEAD_DIM, HIDDEN),
        "self_attn.o_proj.weight": r(HIDDEN, N_HEADS * HEAD_DIM),
        "mlp.gate.weight": r(N_EXP, HIDDEN),
    }


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(11)
    h = torch.randn(BATCH, SEQ, HIDDEN, generator=g) * 0.5
    ang = torch.arange(SEQ * HEAD_DIM, dtype=torch.float32).reshape(
        SEQ, HEAD_DIM) * 0.1
    gu = torch.randn(N_EXP, 2 * INTER, HIDDEN, generator=g) * 0.2
    dn = torch.randn(N_EXP, HIDDEN, INTER, generator=g) * 0.2
    return (h.to(torch.float16), ang.cos().to(torch.float16),
            ang.sin().to(torch.float16), gu.to(torch.float16),
            dn.to(torch.float16))


def _causal():
    m = torch.full((1, 1, SEQ, SEQ), float("-inf"), dtype=torch.float16)
    return torch.triu(m, diagonal=1)


@pytest.mark.parametrize("use_mask", [True, False],
                         ids=["causal_mask", "no_mask"])
def test_the_vulkan_layer_matches_pytorch(weights, data, use_mask):
    hidden, cos, sin, gu, dn = data
    layer = Qwen3LayerWeights(weights, head_dim=HEAD_DIM)
    mask = _causal() if use_mask else None
    got = qwen3_layer_forward_vk(hidden, layer, cos, sin, mask, gu, dn, top_k=TOP_K)
    want = reference(hidden, weights, cos, sin, mask, gu, dn)
    assert got.shape == want.shape, (got.shape, want.shape)
    err = (got.float() - want).abs().max().item()
    # Measured: 0.028 absolute on outputs whose magnitude reaches 14, so about
    # 0.2% - what fp16 arithmetic costs, not what a wrong kernel costs. A
    # relative bound is the honest one: an absolute threshold quietly stops
    # meaning anything if the fixture is ever scaled up.
    scale = want.abs().max().item()
    assert err / scale < 0.01, (
        f"max abs error {err:.4f} on outputs of magnitude {scale:.3f}, "
        f"which is {100 * err / scale:.2f}%")


def test_the_mask_actually_changes_the_answer(weights, data):
    """Otherwise the comparison above passes for the wrong reason."""
    hidden, cos, sin, gu, dn = data
    layer = Qwen3LayerWeights(weights, head_dim=HEAD_DIM)
    a = qwen3_layer_forward_vk(hidden, layer, cos, sin, _causal(), gu, dn, top_k=TOP_K)
    b = qwen3_layer_forward_vk(hidden, layer, cos, sin, None, gu, dn, top_k=TOP_K)
    assert (a.float() - b.float()).abs().max().item() > 1e-3


