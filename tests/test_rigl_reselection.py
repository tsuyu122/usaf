"""--reselect-every has to reselect.

It did not. The universal CLI parsed the number, printed "RigL: every N steps",
wrote it into the checkpoint, and then never used it: RESELECT_EVERY was read
once and referenced once more, to record itself. Only the standalone root
train.py had a do_reselect(). A flag that announces a mechanism and does not
run it is the same defect as the four dead flags found earlier, in the feature
the README names as the adaptive part of "adaptive" fine-tuning.

This is a behavioural test on purpose. One that only checked the parser would
have passed against the broken version.
"""
import os
import re
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
            "--frac", "0.1",
            "--train-from", "0",
            "--eval-every", "0",
            "--save-every", "0",
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


def test_reselect_fires_on_schedule_and_keeps_the_budget(tmp_path):
    out = _run(tmp_path, "--steps", "6", "--reselect-every", "2", "--tag", "r")
    assert "Traceback (most recent" not in out, out[-800:]

    fired = re.findall(r"\[reselect step (\d+)\]", out)
    assert fired, "no reselection happened"
    assert fired == ["2", "4", "6"], f"wrong schedule: {fired!r}"

    reports = re.findall(
        r"\[reselect\] kept=([\d,]+) dropped=([\d,]+) grown=([\d,]+) ", out
    )
    assert len(reports) == len(fired)
    for _kept, dropped, grown in reports:
        # whatever the merge evicts it has to replace, or the budget drifts
        assert dropped == grown, "the merge did not balance"


def test_the_default_interval_does_not_fire_in_a_short_run(tmp_path):
    out = _run(tmp_path, "--steps", "4", "--tag", "n")
    assert "Traceback (most recent" not in out, out[-800:]
    assert not re.findall(r"\[reselect step", out), "fired at the default interval"


def test_the_optimizer_step_counter_survives_a_reselection(tmp_path):
    """Rebuilding the optimizer at every reselect rewound the step counter, and
    the checkpoint recorded the rewound number, so resuming a run that had
    reselected announced a step it was not at.
    """
    import torch

    out = _run(tmp_path, "--steps", "4", "--reselect-every", "2",
               "--save-every", "4", "--tag", "s")
    assert "Traceback (most recent" not in out, out[-800:]

    ckpt = tmp_path / "sparse_step-4.pt"
    assert ckpt.exists(), out[-500:]
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    opt_step = state["optimizer"]["step"]
    assert opt_step == state["step"], (
        f"run ended on step {state['step']}, optimizer recorded {opt_step}"
    )
