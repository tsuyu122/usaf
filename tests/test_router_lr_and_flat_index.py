import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usaf.checkpoint import apply_trained_entries  # noqa: E402
from usaf.train import router_learning_rate  # noqa: E402


def a_gate() -> torch.Tensor:
    """A router as Granite ships it: 32 experts, entries two orders below 1."""
    g = torch.Generator().manual_seed(0)
    return (torch.randn(32, 1024, generator=g) * 0.006).requires_grad_(True)


def _run_900_steps(lr: float) -> tuple[float, float, float]:
    """Move a Granite-shaped gate for a whole run at the given rate.

    Reports how far it moved relative to how big it was to begin with, which
    is the number that decides whether the gate is still a gate.
    """
    g = a_gate()
    base = float(g.detach().pow(2).mean().sqrt())
    opt = torch.optim.SGD([g], lr=lr, momentum=0.9)
    # A constant-magnitude signal, which is the shape of a real training
    # gradient: it has a direction and a size. It does not shrink the weight
    # toward zero, which is what an objective built on the squared norm does,
    # and which is why the first version of the control below could not
    # reproduce the failure it was written to reproduce.
    signal = torch.full_like(g, 0.01)
    before = g.detach().clone()
    for _ in range(900):
        opt.zero_grad()
        g.grad = signal.clone()
        opt.step()
    moved = float((g.detach() - before).pow(2).mean().sqrt())
    return base, moved, moved / base


def test_the_router_moves_by_a_hundredth_of_what_the_experts_do():
    class C:
        lr_peak = 3e-4
        router_lr = 0.0

    assert router_learning_rate(3e-4, 0.0) == pytest.approx(3e-6)
    assert router_learning_rate(3e-4, 0.0) * 100 == pytest.approx(3e-4)


def test_an_explicit_router_lr_is_honoured():
    class C:
        lr_peak = 3e-4
        router_lr = 1e-5

    assert router_learning_rate(3e-4, 1e-5) == 1e-5


def test_the_router_rate_leaves_the_gate_the_size_it_was():
    # 3e-4 per step on weights of 0.006 is a step fifty times the weight. Nine
    # hundred of them grew a real gate from rms 0.0057 to 0.9041, 158x, and the
    # model answered in noise while its training loss fell to 0.001 - the softmax
    # of a huge gate still sums to one, so the loss had nothing to complain
    # about. A hundredth is the difference between tuning a gate and replacing
    # it.
    _, _, ratio = _run_900_steps(3e-6)
    assert ratio < 1.0, ratio


def test_the_expert_rate_would_have_replaced_the_gate():     # The control, so the test above cannot pass for a reason that has nothing     # to do with the fix: the same 900 steps at the rate the experts use.     #     # It moves the gate to 4.4x its own size, and the real run moved one to     # 158x - the toy understates it because its signal is a fixed 0.01, while the     # real gradient was larger than the weights it was pushing. The threshold is     # set at what this model actually demonstrates rather than at the number that     # would be convenient.     _, _, ratio = _run_900_steps(3e-4)     assert ratio > 2.0, ratio   def test_active_idx_is_a_position_in_the_whole_expert_tensor():
    # The bug that cost the export: an index into the flattened [32, 1024, 1024],
    # indexed against the unflattened [32, ...] tensor instead. The measured
    # range is what settles it - 33554426 against a flattened size of 33554432
    # is the last slot of the last expert, and no per-expert index reaches it.
    t = torch.zeros(32, 1024, 1024)
    aidx = torch.tensor([6850, 24260437, 33554431])
    trained = torch.tensor([1.0, 2.0, 3.0])
    out = apply_trained_entries(t, aidx, trained, "x")
    assert out.shape == t.shape
    assert float(out.reshape(-1)[24260437]) == 2.0
    assert float(out.reshape(-1)[33554431]) == 3.0
    assert int(out[0].count_nonzero()) == 1


def test_an_index_outside_the_flat_tensor_is_refused_by_name():
    t = torch.zeros(4, 8, 8)
    with pytest.raises(ValueError, match="fora|outside"):
        apply_trained_entries(t, torch.tensor([999]), torch.tensor([1.0]), "gate")


def test_mismatched_lengths_are_refused():
    t = torch.zeros(4, 8, 8)
    with pytest.raises(ValueError, match="entradas|entries"):
        apply_trained_entries(t, torch.tensor([1, 2]), torch.tensor([1.0]), "gate")


def test_the_router_optimizer_is_built_with_that_rate():
    # The rule being right is worth nothing if the optimizer never asks for it.
    # This run had the rule in the same file as the call that ignored it, and a
    # test of the rule alone passed the whole time - so the wiring is asserted
    # here, against the source that does the wiring.
    import inspect

    import usaf.train as train_module
    src = inspect.getsource(train_module)
    assert "lr=ROUTER_LR" in src
    assert "ROUTER_LR = router_learning_rate(" in src

    # And the experts keep the full rate: a hundredth is for the gate only, and
    # quietly slowing both would turn this into a different run.
    assert "SparseAdam(masters, active_idx=active_idx, lr=LR_PEAK" in src

