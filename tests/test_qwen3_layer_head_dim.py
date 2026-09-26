"""The head count a Qwen3 layer runs with has to come from the model, not a
constant.

Qwen3LayerWeights reported head_dim as the literal 128, and derived num_heads
and num_kv_heads by dividing the projection widths by it. That is right for
Qwen3-30B-A3B and wrong for everything else, and wrong in the worst way:

head_dim cannot be recovered from these tensors at all. q_proj is [num_heads *
head_dim, hidden], so every divisor is consistent with every head count. A model
with head_dim 256 - ZAYA1-8B is hidden 2048 with 8 heads, so exactly 256 - was
reported as 16 heads of 128. Every reshape still worked, because 2048 = 16 *
128 just as well as 8 * 256, and the attention was computed over twice as many
heads as the model has. Nothing raises; the answer is just a different number.

head_dim is a constructor argument now, and it is checked against the
projections: a head_dim that does not divide them is rejected rather than
quietly producing a head count nobody asked for.
"""
import pytest
import torch

from usaf.qwen3_layer_vk import Qwen3LayerWeights


def _w(hidden, n_heads, n_kv, head_dim):
    return {
        "input_layernorm.weight": torch.ones(hidden, dtype=torch.float16),
        "post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.float16),
        "self_attn.q_norm.weight": torch.ones(head_dim, dtype=torch.float16),
        "self_attn.k_norm.weight": torch.ones(head_dim, dtype=torch.float16),
        "self_attn.q_proj.weight": torch.zeros(n_heads * head_dim, hidden,
                                              dtype=torch.float16),
        "self_attn.k_proj.weight": torch.zeros(n_kv * head_dim, hidden,
                                              dtype=torch.float16),
        "self_attn.v_proj.weight": torch.zeros(n_kv * head_dim, hidden,
                                              dtype=torch.float16),
        "self_attn.o_proj.weight": torch.zeros(hidden, n_heads * head_dim,
                                              dtype=torch.float16),
    }


def test_zaya_shape_reports_eight_heads_not_sixteen():
    # ZAYA1-8B: hidden 2048, 8 attention heads, 2 kv heads, so 256 per head.
    weights = Qwen3LayerWeights(_w(2048, 8, 2, 256), head_dim=256)
    assert weights.head_dim == 256
    assert weights.num_heads == 8, weights.num_heads
    assert weights.num_kv_heads == 2, weights.num_kv_heads


def test_the_shape_alone_cannot_recover_head_dim_so_the_default_is_unchanged():
    # The same tensors, no head_dim given: 128 still divides 2048, so the old
    # answer is still reachable - which is why the default is kept and the
    # caller has to say what the model actually is.
    weights = Qwen3LayerWeights(_w(2048, 8, 2, 256))
    assert weights.num_heads == 16


def test_a_head_dim_that_does_not_divide_the_projection_is_refused():
    with pytest.raises(ValueError, match="not a multiple of head_dim"):
        Qwen3LayerWeights(_w(64, 4, 2, 16), head_dim=30)


def test_the_30b_shape_still_works_with_the_default():
    weights = Qwen3LayerWeights(_w(2048, 16, 2, 128))
    assert weights.num_heads == 16
    assert weights.num_kv_heads == 2

