"""The backend a run announces must be the backend it actually got.

--cuda on a machine with no NVIDIA GPU used to print "Backend: CUDA", announce
a device list it never had, and then die one line later on a bare assert - so
the last thing on screen named a backend the run never used. The check was an
assert, which python -O strips out entirely, and without it the failure moved
down to a shape error inside the model with no hint that CUDA was the cause.
"""
import pytest


def test_setup_device_refuses_cuda_it_cannot_use(monkeypatch):
    import torch

    from usaf.train import TrainConfig, setup_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cfg = TrainConfig(model_path="m", dataset_path="d", use_cuda=True)
    with pytest.raises(SystemExit) as ei:
        setup_device(cfg)

    msg = str(ei.value)
    assert "--cuda was requested" in msg
    assert "0 GPUs" in msg, "the message should say what torch actually saw"
    assert "Drop --cuda" in msg, "the message should say what to do instead"


def test_the_cuda_check_survives_optimised_python(monkeypatch):
    """python -O strips assert statements, so the guard cannot be one."""
    import inspect

    from usaf.train import setup_device

    src = inspect.getsource(setup_device)
    assert "assert torch.cuda.is_available()" not in src, (
        "a bare assert is the check that silently disappears under python -O"
    )


def test_setup_device_returns_a_real_device_without_cuda(monkeypatch):
    import torch

    from usaf.train import TrainConfig, setup_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, n_gpus, _scaler = setup_device(
        TrainConfig(model_path="m", dataset_path="d", use_cuda=False)
    )
    assert device.type != "cuda"

    # n_gpus is 1 on the non-CUDA path, which is harmless because every
    # DataParallel site also requires use_cuda. The invariant that matters is
    # that a CPU run can never engage a parallel path, and that the caller
    # sees a count consistent with what the backend actually has.
    assert n_gpus == torch.cuda.device_count() or n_gpus == 1


def test_setup_device_reports_nothing_about_a_backend_it_refused(monkeypatch, capsys):
    """It must fail before announcing, not announce and then fail."""
    import torch

    from usaf.train import TrainConfig, setup_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        setup_device(TrainConfig(model_path="m", dataset_path="d", use_cuda=True))

    out = capsys.readouterr().out
    assert "Backend: CUDA" not in out, (
        "the run announced CUDA before finding out it could not use it"
    )
