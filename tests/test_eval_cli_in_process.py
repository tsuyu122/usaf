"""The universal CLI, run in this process, on a real model and a real checkpoint.

eval_cli was 0% covered because every path into it went through a subprocess.
It is the entry point the final run is meant to use, and the part of it with a
failure mode worth having is the checkpoint: the base model and the fine-tuned
one produce different perplexities, and if the checkpoint is not applied the
number belongs to the base model while the output claims otherwise. That has
been broken before - the key comparison prepended a dot to an already fully
qualified name, so no key could match and the loop fell through.

So the test is: put a known value into a known position of a known parameter,
run the CLI, and look at that position in the model the CLI actually evaluated.
"""
import contextlib
import io
import json
import os
import sys

import pytest
import torch

E2E = r"C:\Users\hm\Projects\e2e"
MODEL = os.path.join(E2E, "tiny-moe")

pytestmark = pytest.mark.skipif(not os.path.isdir(MODEL),
                                reason="e2e fixture absent")


def _run(argv):
    from usaf.eval_cli import main
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["usaf.eval_cli"] + argv
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            main()
    finally:
        sys.argv = old
    return buf.getvalue()


def _first_param_name():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    from transformers import AutoModelForCausalLM
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    for name, p in m.named_parameters():
        if p.numel() >= 8:
            return name, p.numel()
    raise AssertionError("no parameter big enough")


def test_a_checkpoint_without_a_model_is_refused_with_a_reason(tmp_path):
    ck = tmp_path / "c.pt"
    torch.save({"active_idx": {}, "masters": {}}, ck)
    out = _run(["--checkpoint", str(ck)])
    assert "--model is required" in out, out[-1500:]


def test_a_prompt_without_a_model_is_refused_with_a_reason():
    out = _run(["--prompt", "ola"])
    assert "--prompt needs --model" in out, out[-1500:]


def test_the_checkpoint_reaches_the_model_that_was_evaluated(tmp_path,
                                                           monkeypatch):
    """The whole point. Not that it ran - that the value is there after.

    The model has to be the one the CLI held while it evaluated, not a fresh
    load from disk: the checkpoint is applied to the loaded object in memory,
    so a reloaded copy would show the base weights and the test would pass for
    the wrong reason.
    """
    from usaf import eval_cli as cli
    from usaf.eval.benchmark import BenchmarkResults
    name, numel = _first_param_name()
    aidx = torch.tensor([0, 3, 7])
    trained = torch.tensor([9.0, 8.0, 7.0], dtype=torch.float16)
    ck = tmp_path / "c.pt"
    torch.save({"active_idx": {name: aidx},
                "masters": {name: trained},
                "step": 3}, ck)
    seen = []

    def spy(model, tokenizer, device, config, model_name=""):
        seen.append(model)
        return BenchmarkResults(config=config, results={}, model_name=model_name)

    # Patched on the CLI module, not on transformers: eval_cli imports
    # AutoModelForCausalLM inside the function, and transformers' lazy module
    # ignores setattr, so a patch there silently does nothing. run_benchmark is
    # imported at the top and hands over the exact object that was evaluated.
    monkeypatch.setattr(cli, "run_benchmark", spy)
    out = _run(["--model", MODEL, "--checkpoint", str(ck),
                "--datasets", "synthetic-cpp", "--max-samples", "2",
                "--seq-len", "32", "--report", str(tmp_path / "r.json")])
    assert "Traceback" not in out, out[-2500:]
    assert "Checkpoint applied to 1 tensors" in out, out[-2500:]
    assert seen, "the CLI never loaded a model"
    got = dict(seen[0].named_parameters())[name].data.reshape(-1)
    for i, want in zip(aidx.tolist(), trained.tolist()):
        assert float(got[i]) == pytest.approx(want, abs=1e-2), (
            f"{name}[{i}] is {float(got[i])}, the checkpoint said {want}")


def test_a_checkpoint_that_matches_nothing_is_refused(tmp_path):
    ck = tmp_path / "c.pt"
    torch.save({"active_idx": {"not.a.real.parameter": torch.tensor([0])},
                "masters": {"not.a.real.parameter": torch.tensor([1.0])}}, ck)
    with pytest.raises(KeyError, match="do not match the model"):
        _run(["--model", MODEL, "--checkpoint", str(ck),
              "--datasets", "synthetic-cpp", "--max-samples", "2",
              "--seq-len", "32"])


def test_the_report_it_writes_can_be_read_back(tmp_path):
    """A report nothing can load is not a report."""
    r = tmp_path / "r.json"
    _run(["--model", MODEL, "--datasets", "synthetic-cpp",
          "--max-samples", "2", "--seq-len", "32",
          "--report", str(r)])
    assert r.exists()
    data = json.loads(r.read_text(encoding="utf-8"))
    assert "results" in data and "config" in data, list(data)

    from usaf.eval.benchmark import BenchmarkConfig
    # eval_cli rebuilds the config with **data["config"], so anything the
    # dataclass does not accept turns --compare into a TypeError.
    BenchmarkConfig(**data["config"])
