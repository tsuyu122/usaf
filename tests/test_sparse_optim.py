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
