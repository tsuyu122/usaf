"""Resuming a finished run has to still export.

It did not. A run resumed at or past its final step prints "nothing to train"
and returns, and that return sat above the export block. So training on one
machine, coming back later to produce the merged model, wrote no file and
raised nothing. The second command of the ordinary workflow returned success
and did nothing.

The test also checks what came out: a merged file whose active weights match
the trained master to within requantization, and differ from the original
quantized file by much more. An export that happened to be the untouched base
model would pass a "does the file exist" check.
"""
import os
import subprocess
import sys

import pytest

E2E = "C:/Users/hm/Projects/e2e"
MODEL = "tiny-moe"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(E2E, MODEL)),
    reason="e2e fixtures absent",
)


def _run(tmp_path, *extra):
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "usaf.train",
            "--model", MODEL,
            "--dataset", "data.jsonl",
            "--seq-len", "32",
            "--microbatch", "2",
            "--frac", "0.2",
            "--train-from", "0",
            "--eval-every", "0",
            "--lr", "5e-3",
            "--checkpoint-dir", str(tmp_path),
            "--log-dir", str(tmp_path / "logs"),
        ]
        + list(extra),
        cwd=E2E,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (out.stdout or "") + (out.stderr or "")


def test_resuming_a_finished_run_still_exports(tmp_path):
    out = _run(tmp_path, "--steps", "6", "--save-every", "6", "--tag", "t")
    assert "Traceback (most recent" not in out, out[-800:]
    ckpt = tmp_path / "sparse_step-6.pt"
    assert ckpt.exists(), out[-500:]

    target = tmp_path / "merged.pt"
    out = _run(
        tmp_path, "--steps", "6", "--save-every", "0",
        "--resume", str(ckpt), "--export", str(target), "--tag", "e",
    )
    assert "Traceback (most recent" not in out, out[-800:]
    assert "nothing to train" in out, out[-400:]
    assert target.exists(), (
        "resuming a finished run produced no export: " + out[-400:]
    )


def test_the_export_carries_the_trained_weights_not_the_base(tmp_path):
    import torch

    from usaf.quantization import dequantize_4bit

    def deq(e):
        if isinstance(e, dict) and "q" in e:
            return dequantize_4bit(e["q"], e["s"], e["z"], e["shape"], group_size=128)
        return dequantize_4bit(e[0], e[1], e[2], e[3], group_size=128)

    out = _run(tmp_path, "--steps", "20", "--save-every", "20", "--tag", "t")
    assert "Traceback (most recent" not in out, out[-800:]
    ckpt = tmp_path / "sparse_step-20.pt"
    assert ckpt.exists(), out[-500:]

    target = tmp_path / "merged.pt"
    out = _run(
        tmp_path, "--steps", "20", "--save-every", "0",
        "--resume", str(ckpt), "--export", str(target), "--tag", "e",
    )
    assert target.exists(), out[-400:]

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    exported = torch.load(target, map_location="cpu", weights_only=False)
    original = torch.load(
        os.path.join(E2E, MODEL + "-q4", "experts_q4.pt"),
        map_location="cpu", weights_only=False,
    )

    name = sorted(state["masters"])[0]
    d = deq(exported[name]).reshape(-1).float()
    o = deq(original[name]).reshape(-1).float()
    aidx = state["active_idx"][name].reshape(-1).to(torch.long)
    m = state["masters"][name].reshape(-1).float()

    drift = float((d - o).abs().max())
    err_export = float((d[aidx] - m).abs().max())
    err_base = float((o[aidx] - m).abs().max())

    assert drift > 0.0, "the export is byte-identical to the base model"
    assert err_export < err_base / 3.0, (
        "the export is no closer to the trained weights than the base is: "
        "%g vs %g" % (err_export, err_base)
    )
