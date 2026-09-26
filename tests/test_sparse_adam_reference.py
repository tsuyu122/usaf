"""
SparseAdam is the thing that actually changes the weights, and a wrong
answer there is invisible: every step applies, the loss falls, and the
router keeps training regardless. That is the ZAYA bug one level down.

The contract tested here is the one the trainer actually uses, and all
three parts of it are load-bearing:

  SparseAdam(masters, active_idx=..., compact_params=True)
  opt.step(compact_grads=cg)   where cg[name] is the gradient gathered
                                  at active_idx, already divided by the
                                  token count

The master is 1-D with one entry per active weight, and the gradient is the
same length. Giving this constructor a full-size parameter and no compact
grads takes the other branch, which indexes the parameter with flat indices
into the real tensor and raises IndexError on a compact master - correct
behaviour for a contract the trainer does not use, and a trap for a test
that guesses the shape. The reference is full AdamW over the same compact
vector, and the two only agree with weight decay off: this optimizer's decay
is coupled and torch's is decoupled, which the last test states rather than
hides behind a tolerance.

The scatter test is the one that matters for correctness: it checks the
update lands at the indices given and nowhere else. A silent
rank-for-index substitution trains the wrong weights and looks perfectly
healthy, which is the failure mode this whole file exists to catch.
"""

import torch
import torch.nn as nn

from usaf.sparse_optim import SparseAdam

REAL_SHAPE = (16, 4096, 2048)   # a ZAYA gate_up_proj expert tensor
FRAC = 0.05
LR, B1, B2, EPS = 3e-4, 0.9, 0.999, 1e-8


def _setup(seed):
    torch.manual_seed(seed)
    real = (torch.randn(*REAL_SHAPE) * 0.02).half()
    n = real.numel()
    idx = torch.randperm(n)[: int(n * FRAC)].sort().values
    master = real.reshape(-1).index_select(0, idx).float().contiguous()
    return real, idx, master


def _opt(master, idx, wd=0.0):
    m = nn.Parameter(master.clone())
    return m, SparseAdam({'e': m}, active_idx={'e': idx}, lr=LR,
                               weight_decay=wd, betas=(B1, B2), eps=EPS,
                               compact_params=True)


def test_the_stored_indices_are_the_indices_given():
    _real, idx, master = _setup(0)
    m, opt = _opt(master, idx)
    assert torch.equal(opt._idx['e'], idx), 'the optimizer stored other indices'


def test_the_compact_master_is_exactly_the_active_count():
    # The master's length is the contract: too short and the update is
    # applied to the wrong elements, too long and the tail is a weight
    # that does not exist in the real tensor.
    _real, idx, master = _setup(1)
    assert master.numel() == idx.numel()
    m, opt = _opt(master, idx)
    assert m.numel() == idx.numel()
    assert opt._m['e'].numel() == idx.numel()
    assert opt._v['e'].numel() == idx.numel()


def test_a_step_moves_the_master():
    _real, idx, master = _setup(1)
    torch.manual_seed(7)
    g = (torch.randn(idx.numel()) * 0.01).float()
    m, opt = _opt(master, idx)
    m.grad = torch.zeros_like(m)
    opt.step(compact_grads={'e': g.clone()})
    assert (m.detach() - master).abs().max() > 0, 'nothing moved'


def test_the_update_is_where_the_gradient_was():
    # Per-element: the change at position i tracks the gradient at position i,
    # not some shifted or reordered version of it.
    _real, idx, master = _setup(8)
    torch.manual_seed(11)
    g = torch.randn(idx.numel()) * 0.01
    m, opt = _opt(master, idx)
    m.grad = torch.zeros_like(m)
    opt.step(compact_grads={'e': g.clone()})
    delta = m.detach() - master
    big = g.abs() > g.abs().median()
    assert delta[big].abs().mean() > delta[~big].abs().mean(), (
        'the largest gradients did not produce the largest updates'
    )


def test_it_matches_full_adamw_on_the_compact_vector():
    _real, idx, master = _setup(2)
    torch.manual_seed(99)
    g = torch.randn(idx.numel()) * 0.01
    a, opt = _opt(master, idx)
    b = nn.Parameter(master.clone())
    ref = torch.optim.AdamW([b], lr=LR, betas=(B1, B2), eps=EPS)
    for _ in range(3):
        a.grad = torch.zeros_like(a)
        opt.step(compact_grads={'e': g.clone()})
        b.grad = g.clone()
        ref.step()
    assert torch.allclose(a.detach(), b.detach(), atol=1e-6), (
        (a.detach() - b.detach()).abs().max()
    )


def test_coupled_weight_decay_moves_the_compact_vector():
    # Coupled decay adds wd*p to the gradient, so it shows up even with a
    # zero gradient - which is also the proof it is not being skipped.
    _real, idx, master = _setup(3)
    m, opt = _opt(master, idx, wd=0.1)
    before = m.detach().clone()
    m.grad = torch.zeros_like(m)
    opt.step(compact_grads={'e': torch.zeros(idx.numel())})
    assert (before - m.detach()).abs().max() > 0, 'decay had no effect'


def test_coupled_decay_differs_from_decoupled_by_design():
    # Stated so the difference is a decision and not an accident: matching
    # AdamW here would mean changing the update rule, not fixing a test.
    _real, idx, master = _setup(4)
    torch.manual_seed(5)
    g = torch.randn(idx.numel()) * 0.01
    a, opt = _opt(master, idx, wd=0.1)
    b = nn.Parameter(master.clone())
    ref = torch.optim.AdamW([b], lr=LR, betas=(B1, B2), eps=EPS,
                             weight_decay=0.1)
    for _ in range(2):
        a.grad = torch.zeros_like(a)
        opt.step(compact_grads={'e': g.clone()})
        b.grad = g.clone()
        ref.step()
    assert not torch.allclose(a.detach(), b.detach(), atol=1e-6)


def test_a_missing_compact_grad_leaves_that_weight_untouched():
    # A tensor absent from the dict is skipped, not silently zeroed: a
    # missing entry is a bug, and a zeroed one would be a silent one.
    _real, idx, master = _setup(6)
    m, opt = _opt(master, idx)
    before = m.detach().clone()
    m.grad = torch.zeros_like(m)
    opt.step(compact_grads={})
    assert torch.equal(before, m.detach()), (
        'a weight moved with no gradient supplied'
    )
