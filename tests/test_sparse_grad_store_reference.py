"""
The compact gradient is what the optimizer is handed, and its scale is the
learning rate.

The trainer divides the store by loss_scale * ACCUM, and a store that
accumulates over tokens and microbatches has to sum into exactly the
gradient a plain backward would have produced, gathered at the active
indices. If it double-counted, or missed a token, or gathered the wrong
positions, the loss would still fall - a wrong scale is a wrong learning
rate, not a crash, and a smaller one looks like a model converging slowly.

So this compares SparseGradStore against autograd on the same loss. The
module routes through the expert forward that installs the capture hooks,
because testing the arithmetic directly would prove the maths and miss the
protocol, which is where the naming and the per-expert mapping live.
"""

import pytest
import torch
import torch.nn as nn

from usaf.batched_experts import batched_or_loop_forward
from usaf.moe_loader import SparseGradStore

NEXP, HID, INTER, NTOK = 4, 8, 6, 5
PREFIX = 'model.layers.0.mlp.experts'
PARAM = PREFIX + '.gate_up_proj'


class Experts(nn.Module):
    def __init__(self):
        torch.manual_seed(0)
        super().__init__()
        self.num_experts = NEXP
        self.gate_up_proj = nn.Parameter(torch.randn(NEXP, 2 * INTER, HID) * 0.2)
        self.down_proj = nn.Parameter(torch.randn(NEXP, HID, INTER) * 0.2)
        self.act_fn = nn.SiLU()

    def forward(self, x, w, top_k_index=None, top_k_weights=None, *, dense_weights=None):
        n = self.num_experts
        gu = torch.bmm(x.expand(n, *x.shape), self.gate_up_proj.transpose(1, 2))
        gate, up = gu.chunk(2, dim=-1)
        cur = self.act_fn(gate) * up
        out = torch.bmm(cur, self.down_proj.transpose(1, 2))
        ww = dense_weights.t().to(out.dtype).unsqueeze(-1)
        return (out * ww).sum(0)


@pytest.fixture
def experts():
    return Experts()


@pytest.fixture
def inputs():
    torch.manual_seed(3)
    x = torch.randn(NTOK, HID)
    w = torch.rand(NTOK, NEXP) + 0.1
    w = w / w.sum(-1, keepdim=True)
    return x, w


def _store(experts, idx):
    return SparseGradStore({PARAM: idx},
                           {PARAM: tuple(experts.gate_up_proj.shape)})


def _captured(experts, x, w, idx, backwards=1):
    # Runs the loss through the real capture path and returns the store.
    store = _store(experts, idx)
    cls = type(experts)
    saved = cls.forward
    cls.forward = batched_or_loop_forward
    experts._grad_capture = (store, PREFIX)
    try:
        loss = experts(x, None, None, dense_weights=w).pow(2).mean()
        for _ in range(backwards - 1):
            loss.backward(retain_graph=True)
        loss.backward()
    finally:
        cls.forward = saved
        del experts._grad_capture
    return store


def test_the_store_matches_a_plain_backward(experts, inputs):
    x, w = inputs
    idx = torch.randperm(experts.gate_up_proj.numel())[:12].sort().values

    want = experts(x, None, None, dense_weights=w).pow(2).mean()
    want.backward()
    want = experts.gate_up_proj.grad.detach().reshape(-1).index_select(0, idx)

    experts.zero_grad(set_to_none=True)
    got = _captured(experts, x, w, idx).compact[PARAM]
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_two_backwards_accumulate_to_twice_the_gradient(experts, inputs):
    x, w = inputs
    idx = torch.randperm(experts.gate_up_proj.numel())[:12].sort().values
    experts(x, None, None, dense_weights=w).pow(2).mean().backward()
    g = experts.gate_up_proj.grad.detach().reshape(-1).index_select(0, idx)
    want = g * 2
    experts.zero_grad(set_to_none=True)
    got = _captured(experts, x, w, idx, backwards=2).compact[PARAM]
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_zero_clears_the_accumulator(experts):
    idx = torch.randperm(experts.gate_up_proj.numel())[:12].sort().values
    store = _store(experts, idx)
    store.add(PARAM, 0, torch.ones(2 * INTER * HID))
    assert store.compact[PARAM].abs().max() > 0
    store.zero_()
    assert store.compact[PARAM].abs().max() == 0, (
        'the second step would add to the first step gradients'
    )


def test_an_expert_with_no_active_weight_is_a_no_op(experts):
    # Active weights only in expert 0. A hook firing for expert 1 has nowhere
    # to land and must be dropped, not written into another expert's slots.
    slice_size = 2 * INTER * HID
    store = _store(experts, torch.arange(0, 5))
    store.add(PARAM, 1, torch.ones(slice_size))
    assert store.compact[PARAM].abs().max() == 0, (
        'a gradient arrived for an expert with no active weights'
    )
    store.add(PARAM, 0, torch.ones(slice_size))
    assert store.compact[PARAM].abs().max() > 0


def test_the_gather_uses_the_right_expert_slice(experts):
    # The map splits flat indices by expert and subtracts the slice base. A
    # wrong base gathers another expert's weights and nothing raises.
    slice_size = 2 * INTER * HID
    in_e2 = torch.arange(0, slice_size, dtype=torch.long)[:6]
    idx = torch.cat([torch.arange(0, 4), 2 * slice_size + in_e2])
    store = _store(experts, idx)
    e2 = experts.gate_up_proj.detach()[2].reshape(-1)[in_e2].clone()
    store.add(PARAM, 2, e2)
    assert torch.equal(store.compact[PARAM][-6:], e2), (
        'the gathered values belong to a different expert'
    )
