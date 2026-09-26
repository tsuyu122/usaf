"""An update to a non-contiguous parameter used to be written to a throwaway copy.

reshape(-1) returns a view only when the tensor is already contiguous. On one
that is not, it copies, and the scatter_ that followed updated the copy. The
parameter came out bit-identical, with no error and no warning, while the loss
kept falling because the router params still moved.

Two things have to hold, and the first fix broke the second: the element has to
change, and the parameter has to keep the shape the next forward expects. An
expert weight is [hidden, inter]; adopting a flattened copy would hand a 1-D
weight to F.linear and the model would be quietly wrong instead of quietly
frozen.

Every expert tensor the existing tests build is contiguous, so the old code was
right for the case that was tried and wrong for the one that was not.
"""
import torch
from torch.nn.functional import linear

from usaf.sparse_optim import SparseAdam


def _a_non_contiguous_expert(rows=3, cols=4):
    # [rows, cols] via a transposed view of a [cols, rows] tensor: the shape
    # an F.linear weight has, and non-contiguous.
    base = torch.arange(rows * cols, dtype=torch.float32).reshape(cols, rows)
    return torch.nn.Parameter(base.t())


def _step(p, idx, grads):
    names = {"e.gate_up_proj": p}
    opt = SparseAdam(
        names, None,
        active_idx={"e.gate_up_proj": torch.tensor(idx, dtype=torch.long)},
        lr=1.0, betas=(0.0, 0.0), eps=0.0, weight_decay=0.0,
    )
    opt.step(compact_grads={"e.gate_up_proj": torch.ones(len(idx))})


def test_the_parameter_in_the_test_is_really_non_contiguous():
    p = _a_non_contiguous_expert()
    assert not p.data.is_contiguous(), "the fixture stopped being the case it tests"
    assert tuple(p.shape) == (3, 4), p.shape


def test_a_non_contiguous_parameter_really_receives_the_adam_update():
    p = _a_non_contiguous_expert()
    before = p.detach().reshape(-1).clone()
    _step(p, [0, 5], None)
    after = p.detach().reshape(-1)
    assert not torch.equal(before, after), (
        "the update went to a copy: the parameter is bit-identical")
    assert (before[0] != after[0]).item(), "index 0 did not move"
    assert (before[5] != after[5]).item(), "index 5 did not move"
    assert (before[1] == after[1]).item(), "an inactive element was touched"


def test_the_update_keeps_the_shape_the_next_forward_expects():
    """The flattened-copy fix moved the numbers and broke the model."""
    p = _a_non_contiguous_expert()
    shape = tuple(p.shape)
    _step(p, [0], None)
    assert tuple(p.shape) == shape, f"{shape} became {tuple(p.shape)}"
    rows, cols = shape
    # linear(input, weight) needs input @ weight.T, so input is [*, cols].
    out = linear(torch.ones(1, cols), p.detach())
    assert tuple(out.shape) == (1, rows), out.shape
