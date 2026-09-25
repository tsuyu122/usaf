"""Tests for SparseAdam optimizer."""
import torch

from usaf.sparse_optim import SparseAdam


def test_sparse_adam_init_with_idx():
    params = {
        "layer1.weight": torch.nn.Parameter(torch.randn(10, 10)),
        "layer2.weight": torch.nn.Parameter(torch.randn(5, 5)),
    }
    active_idx = {
        "layer1.weight": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "layer2.weight": torch.tensor([0, 5, 10], dtype=torch.long),
    }
    opt = SparseAdam(params, active_idx=active_idx, lr=1e-3)
    assert opt.num_active_params == 8
    assert opt.optimizer_memory_mb > 0


def test_sparse_adam_step():
    param = torch.nn.Parameter(torch.ones(10))
    params = {"w": param}
    active_idx = {"w": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)}
    opt = SparseAdam(params, active_idx=active_idx, lr=1.0)
    loss = param[:5].sum()
    loss.backward()
    opt.step()
    assert param[0].item() < 1.0
    assert param[5].item() == 1.0


def test_sparse_adam_step_no_grad():
    params = {"w": torch.nn.Parameter(torch.ones(10))}
    active_idx = {"w": torch.tensor([0, 1], dtype=torch.long)}
    opt = SparseAdam(params, active_idx=active_idx, lr=1.0)
    opt.step()
    assert True


def test_sparse_adam_compact_grads():
    param = torch.nn.Parameter(torch.ones(10))
    params = {"w": param}
    active_idx = {"w": torch.tensor([0, 1, 2], dtype=torch.long)}
    opt = SparseAdam(params, active_idx=active_idx, lr=1.0)
    compact_grads = {"w": torch.ones(3)}
    opt.step(compact_grads=compact_grads)
    assert param[0].item() < 1.0
    assert param[3].item() == 1.0


def test_sparse_adam_zero_grad():
    param = torch.nn.Parameter(torch.randn(10))
    params = {"w": param}
    active_idx = {"w": torch.tensor([0, 1], dtype=torch.long)}
    opt = SparseAdam(params, active_idx=active_idx)
    opt.zero_grad()
    assert True


def test_sparse_adam_state_dict():
    params = {"w": torch.nn.Parameter(torch.ones(10))}
    active_idx = {"w": torch.tensor([0, 1], dtype=torch.long)}
    opt = SparseAdam(params, active_idx=active_idx)
    sd = opt.state_dict()
    assert "step" in sd
    assert "m" in sd
    assert "v" in sd
    assert sd["step"] == 0


def test_sparse_adam_load_state_dict():
    params = {"w": torch.nn.Parameter(torch.ones(10))}
    active_idx = {"w": torch.tensor([0, 1], dtype=torch.long)}
    opt = SparseAdam(params, active_idx=active_idx)
    sd = opt.state_dict()
    sd["step"] = 5
    opt.load_state_dict(sd)
    assert opt._step == 5

def test_sparse_adam_reselect_preserves_momentum():
    """A RigL reselection must not wipe the Adam moments.

    Rebuilding the state from scratch on every reselection resets m and v to
    zero while keeping the step counter, so the bias correction no longer
    matches the moment tensors and the optimizer effectively restarts every
    RESELECT_EVERY steps. Elements that remain active must carry their
    moments across the reselection.
    """
    params = {"w": torch.nn.Parameter(torch.zeros(10))}
    opt = SparseAdam(
        params, active_idx={"w": torch.tensor([1, 2, 3, 4, 5])},
        lr=0.1, weight_decay=0.0,
    )
    for _ in range(3):
        opt.step(compact_grads={"w": torch.full((5,), 0.01)})
    m_before = opt._m["w"].clone()
    step_before = opt._step

    # Keep {2,3,4} from the old set and add two new elements, 7 and 8.
    opt.reselect(params, {"w": torch.tensor([2, 3, 4, 7, 8])})

    assert opt._step == step_before, "step counter was reset by reselect()"
    m_after = opt._m["w"]
    assert torch.allclose(m_after[:3], m_before[1:4]), (
        "surviving elements lost their Adam momentum"
    )
    assert torch.all(m_after[3:] == 0), (
        "newly activated elements should start with zero momentum"
    )
