"""VKLayer.forward_qkv is the Vulkan path the root trainer actually uses.

The root train.py builds these layers and calls forward_qkv to produce Q, K
and V, then swaps the projections out for the results before native DML
 attention. Nothing compared that output against the PyTorch projections it
replaces, so a kernel that transposed a weight the wrong way or mis-sized a
buffer would only have shown up as a model that trained badly. The individual
kernels are pinned in test_vulkan_kernels_equivalence; this pins the composed
path, including the [B*S, out] layout the caller reshapes.
"""
import numpy as np
import pytest

vkl = pytest.importorskip("usaf.vk_layer")

pytestmark = pytest.mark.skipif(not getattr(vkl, "HAS_VK", False),
                                reason="Vulkan module not built")

HIDDEN = 256
HEADS = 8
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS


def _weights(seed=0):
    rng = np.random.default_rng(seed)

    def w(shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float16)

    return {
        "input_layernorm.weight": w((HIDDEN,)),
        "self_attn.q_proj.weight": w((HEADS * HEAD_DIM, HIDDEN)),
        "self_attn.k_proj.weight": w((KV_HEADS * HEAD_DIM, HIDDEN)),
        "self_attn.v_proj.weight": w((KV_HEADS * HEAD_DIM, HIDDEN)),
    }


@pytest.fixture(scope="module")
def layer():
    weights = _weights()
    lay = vkl.VKLayer(HIDDEN, HEADS, KV_HEADS, HEAD_DIM, weights)
    assert lay._uploaded, "__init__ uploads the weights"
    return lay, weights


def _rmsnorm(x, weight, eps=1e-6):
    xf = x.astype(np.float32)
    var = np.mean(xf ** 2, axis=-1, keepdims=True)
    return (xf / np.sqrt(var + eps) * weight.astype(np.float32)).astype(np.float16)


def test_forward_qkv_matches_the_pytorch_projections(layer):
    lay, weights = layer
    rng = np.random.default_rng(1)
    b, s = 2, 6
    hidden = (rng.standard_normal((b, s, HIDDEN)) * 0.5).astype(np.float16)

    q, k, v = lay.forward_qkv(hidden)

    # flattened [B*S, out]; the caller in train.py reshapes for attention
    assert q.shape == (b * s, HEADS * HEAD_DIM)
    assert k.shape == (b * s, KV_HEADS * HEAD_DIM)
    assert v.shape == (b * s, KV_HEADS * HEAD_DIM)

    normed = _rmsnorm(hidden.reshape(b * s, HIDDEN),
                      weights["input_layernorm.weight"])

    for got, key, out_dim in (
        (q, "self_attn.q_proj.weight", HEADS * HEAD_DIM),
        (k, "self_attn.k_proj.weight", KV_HEADS * HEAD_DIM),
        (v, "self_attn.v_proj.weight", KV_HEADS * HEAD_DIM),
    ):
        w = weights[key].astype(np.float32)
        want = (normed.astype(np.float32) @ w.T).astype(np.float16)
        gotf = got.astype(np.float32)
        rel = np.abs(gotf - want.astype(np.float32)).max()
        rel /= max(np.abs(want.astype(np.float32)).max(), 1e-6)
        assert rel < 2e-2, key + " differs by " + format(rel, ".2e")


def test_forward_qkv_refuses_a_layer_whose_weights_are_not_uploaded():
    """A half-built layer has to say so rather than read uninitialised buffers."""
    lay = vkl.VKLayer(HIDDEN, HEADS, KV_HEADS, HEAD_DIM, _weights(2))
    lay._uploaded = False
    with pytest.raises(RuntimeError, match="not uploaded"):
        lay.forward_qkv(np.zeros((1, 2, HIDDEN), np.float16))
