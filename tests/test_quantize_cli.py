"""usaf.quantize closes the loop the training CLI could not close on its own.

The trainer refuses to start without an experts_q4.pt, and nothing in the
project could produce one: the root script loads a file it assumes exists and
the library exposes quantize_state_dict with no way to gather a model and call
it. So "point it at any MoE model" stopped one step short, at exactly the point
where you would find out on a new machine.
"""
import os
import shutil
import subprocess
import sys

import pytest

E2E = "C:/Users/hm/Projects/e2e"
MODEL = "tiny-moe"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(E2E, MODEL)),
    reason="e2e fixtures absent",
)


@pytest.fixture(scope="module")
def quantized(tmp_path_factory):
    """A copy of the fixture, quantized from scratch into a temp dir."""
    work = tmp_path_factory.mktemp("quant")
    model = work / "m"
    shutil.copytree(os.path.join(E2E, MODEL), model)

    out = subprocess.run(
        [sys.executable, "-m", "usaf.quantize", "--model", str(model)],
        cwd=E2E,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = (out.stdout or "") + (out.stderr or "")
    assert "Traceback" not in log, log[-800:]
    produced = work / "m-q4" / "experts_q4.pt"
    assert produced.exists(), log[-600:]
    return model, produced, log


def test_it_produces_the_file_the_trainer_looks_for(quantized):
    _model, produced, _log = quantized
    assert produced.stat().st_size > 0


def test_the_output_actually_compresses(quantized):
    import torch

    _model, produced, log = quantized
    q = torch.load(produced, map_location="cpu", weights_only=False)
    assert q, "the quantized dict is empty"
    for name, entry in q.items():
        assert isinstance(entry, dict) and "q" in entry, name
        assert "s" in entry and "z" in entry and "shape" in entry, name

    raw = 0
    for entry in q.values():
        raw += int(entry['q'].numel()) * 2
    assert raw > 0
    # 4-bit plus a scale and zero per group: about 3.7x, and always under 4x
    assert produced.stat().st_size < raw, "the file is not smaller than fp16"
    assert produced.stat().st_size > raw / 5, "implausibly small"


def test_the_trainer_then_runs_on_it(quantized):
    """The point of the module: the two commands work back to back."""
    model, _produced, _log = quantized
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "usaf.train",
            "--model", str(model),
            "--dataset", "data.jsonl",
            "--seq-len", "32",
            "--microbatch", "2",
            "--frac", "0.1",
            "--train-from", "0",
            "--steps", "3",
            "--eval-every", "0",
            "--save-every", "0",
            "--checkpoint-dir", str(model),
            "--log-dir", str(model / "logs"),
        ],
        cwd=E2E,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = (out.stdout or "") + (out.stderr or "")
    assert "Traceback" not in log, log[-800:]
    assert "Loss:" in log, log[-500:]
    assert "m-q4" in log, "the trainer did not auto-detect the file it needs"


def test_a_model_without_weights_is_refused(tmp_path):
    d = tmp_path / "noweights"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-m", "usaf.quantize", "--model", str(d)],
        cwd=E2E,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = (out.stdout or "") + (out.stderr or "")
    # detect_model refuses this first: the config has no model_type, and a raw
    # ValueError from deep inside configuration_auto would name neither that nor
    # what to do about it.
    assert "is not a config transformers can read" in log, log[-400:]
    assert "Traceback" not in log, log[-400:]
