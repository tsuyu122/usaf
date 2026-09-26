"""
The batched path has to be the same function as the loop it replaces.

Two paths now exist for the same experts: a loop of 2*num_experts small matmuls,
and a pair of batched matmuls. If they disagree, every perplexity, every
generated sample and every frozen-cache activation comes from a different
model than the one trained - and the frozen cache is reused across runs, so
the error would outlive the run that made it.

They are not bit-equal and the test does not ask them to be: the loop adds up
float32 expert by expert, bmm accumulates in the output dtype. The bar is fp16
precision, which is the cost of the representation rather than of the batching.

The dispatcher tests are the ones that matter for training: with a capture
installed it must take the loop, because the hooks attach to the slices the
loop produces. A dispatcher that always chose bmm would be faster and would
train nothing.
"""

import pytest
import torch
import torch.nn as nn

from usaf.batched_experts import (
    batched_or_loop_forward,
    dml_qwen3_experts_forward_batched,
)
from usaf.qwen3moe_dml import dml_qwen3_experts_forward

HIDDEN, INTER, NEXP, NTOK = 16, 8, 5, 12


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = NEXP
        torch.manual_seed(0)
        self.gate_up_proj = nn.Parameter(torch.randn(NEXP, 2 * INTER, HIDDEN) * 0.2)
        self.down_proj = nn.Parameter(torch.randn(NEXP, HIDDEN, INTER) * 0.2)
        self.act_fn = nn.SiLU()


@pytest.fixture
def inputs():
    torch.manual_seed(1)
    hs = torch.randn(NTOK, HIDDEN)
    weights = torch.rand(NTOK, NEXP) + 0.05
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return hs, weights


def _run(fn, mod, hs, weights):
    with torch.no_grad():
        return fn(mod, hs, None, None, dense_weights=weights)


def test_batched_matches_the_loop(inputs):
    mod = Experts()
    hs, weights = inputs
    loop = _run(dml_qwen3_experts_forward, mod, hs, weights)
    batched = _run(dml_qwen3_experts_forward_batched, mod, hs, weights)
    scale = loop.abs().max().clamp_min(1e-6)
    rel = (loop - batched).abs().max() / scale
    assert rel < 0.02, f"""relative disagreement {rel:.4f} is beyond fp16 noise"""


def test_batched_keeps_the_input_shape(inputs):
    mod = Experts()
    hs, weights = inputs
    out = _run(dml_qwen3_experts_forward_batched, mod, hs, weights)
    assert out.shape == hs.shape, f"""{hs.shape} became {out.shape}"""


def test_batched_works_on_a_3d_input(inputs):
    # The MoE block passes [1, seq, hidden]; flattening it wrongly would show
    # up here and nowhere else, because the loop takes the same call.
    mod = Experts()
    hs, weights = inputs
    hs3 = hs.unsqueeze(0)
    out = _run(dml_qwen3_experts_forward_batched, mod, hs3, weights)
    assert out.shape == hs3.shape, f"""{hs3.shape} became {out.shape}"""


def test_the_dispatcher_takes_the_loop_when_capturing(inputs):
    # Faster with no learning is worse than slow learning.
    mod = Experts()
    hs, weights = inputs
    captured = {}
    mod._grad_capture = (captured, """p""")
    try:
        # Hooks registered on the expert slices only fire on a backward, so
        # a forward alone proves nothing about which path ran.
        out = batched_or_loop_forward(
            mod, hs, None, None, dense_weights=weights
        )
        out.sum().backward()
    finally:
        del mod._grad_capture
    # The loop path registers hooks on the expert slices. The bmm path cannot,
    # so a capture that stayed empty is the signature of the wrong path.
    assert captured, (
        """the dispatcher took the batched path while capturing: """
        """the sparse run would receive no gradient"""
    )


def test_the_dispatcher_matches_the_loop_when_not_capturing(inputs):
    mod = Experts()
    hs, weights = inputs
    with torch.no_grad():
        a = batched_or_loop_forward(mod, hs, None, None, dense_weights=weights)
    with torch.no_grad():
        b = dml_qwen3_experts_forward(mod, hs, None, None, dense_weights=weights)
    assert torch.allclose(a, b, atol=1e-3, rtol=1e-2)
