"""SparseAdam has to be Adam.

It reimplements the update by hand over a sparse index set, with its own bias
correction, its own eps placement and a hand-rolled L2 term. Nothing compared
it against torch.optim.Adam, so a transposed bias correction, an eps added
inside the square root, or a decay term applied to the wrong tensor would all
look like a plausible optimizer and produce a subtly worse model.

With every element active the sparse path degenerates to the dense one, which
is what makes this comparison possible: same gradients in, same parameters out.
"""
import pytest
import torch

from usaf.sparse_optim import SparseAdam


def _grads(seed, steps, numel):
    torch.manual_seed(seed)
    return [torch.randn(numel) * (0.1 + 0.9 * i / steps) for i in range(steps)]


def _run_sparse(seed, lr, betas, eps, wd, steps, numel=64):
    grads = _grads(seed, steps, numel)
    p = torch.nn.Parameter(torch.zeros(numel))
    opt = SparseAdam(
        {"w": p},
        active_idx={"w": torch.arange(numel)},
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=wd,
        compact_params=False,
    )
    for g in grads:
        p.grad = g.clone()
        opt.step()
    return p.detach().clone()


def _run_torch(seed, lr, betas, eps, wd, steps, numel=64):
    grads = _grads(seed, steps, numel)
    p = torch.nn.Parameter(torch.zeros(numel))
    opt = torch.optim.Adam([p], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    for g in grads:
        p.grad = g.clone()
        opt.step()
    return p.detach().clone()


@pytest.mark.parametrize("lr", [1e-2, 1e-3])
@pytest.mark.parametrize("wd", [0.0, 0.01])
def test_sparse_adam_matches_torch_adam(lr, wd):
    a = _run_sparse(0, lr, (0.9, 0.999), 1e-8, wd, 12)
    b = _run_torch(0, lr, (0.9, 0.999), 1e-8, wd, 12)
    rel = float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))
    assert rel < 1e-5, "differs from torch.optim.Adam by " + format(rel, ".2e")


def test_bias_correction_is_not_hoisted_out_of_the_loop():
    """On the first step a large gradient must move the parameter by about -lr.

    Hoisting the bias correction out of the step counter would make that
    movement wrong by a factor of 1 - beta1, which is invisible after many
    small steps and plainly wrong on the first one.
    """
    p = torch.nn.Parameter(torch.zeros(4))
    opt = SparseAdam(
        {"w": p},
        active_idx={"w": torch.arange(4)},
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        compact_params=False,
    )
    p.grad = torch.ones(4)
    opt.step()
    # m_hat = 1 and v_hat = 1, so the update is 1 and the step is -lr
    assert torch.allclose(p.detach(), torch.full((4,), -0.1), atol=1e-6)


def test_reselect_keeps_moments_for_surviving_elements():
    """RigL reselects every reselect_every steps.

    Elements that stay active keep their Adam moments, elements newly
    activated start clean, and the step counter must not be rewound or the
    bias correction desynchronises from the moments it is correcting.
    """
    numel = 16
    p = torch.nn.Parameter(torch.zeros(numel))
    opt = SparseAdam(
        {"w": p},
        active_idx={"w": torch.arange(8)},
        lr=1e-2,
        compact_params=False,
    )
    p.grad = torch.arange(numel, dtype=torch.float32)
    opt.step()

    m_before = opt.state_dict()["m"]["w"].clone()
    v_before = opt.state_dict()["v"]["w"].clone()
    step_before = opt.state_dict()["step"]

    # keep 0..5, drop 6..7, add 8..11
    new_idx = torch.tensor([0, 1, 2, 3, 4, 5, 8, 9, 10, 11])
    opt.reselect({"w": p}, {"w": new_idx})

    m_after = opt.state_dict()["m"]["w"]
    v_after = opt.state_dict()["v"]["w"]

    assert torch.allclose(m_after[:6], m_before[:6], atol=1e-7), "survivors lost m"
    assert torch.allclose(v_after[:6], v_before[:6], atol=1e-7), "survivors lost v"
    assert float(m_after[6:].abs().max()) == 0.0, "new elements kept stale moments"
    assert float(v_after[6:].abs().max()) == 0.0
    assert opt.state_dict()["step"] == step_before, "the step counter was rewound"
