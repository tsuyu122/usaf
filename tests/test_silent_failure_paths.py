"""Two ways a run could report success while having done the wrong thing.

Both were found by re-reading rather than by running, and both are the same
shape as the DataParallel one: an exception handler that turns a hard failure
into a quiet continuation.

  * --export is requested, the merge raises, the run prints one line and
    finishes with exit code 0 and "Complete". The export is the entire reason
    for the second command in "train on one machine, export on another", so
    nothing tells you to go looking for the missing file.

  * a model transformers cannot build is replaced by a Qwen3-MoE built from the
    same config. That converts "this transformers cannot load that model" into
    "train a different model" - quietly, and with a loss curve that looks like
    progress. The substitution only ever meant anything for a Qwen3-MoE config.
"""
import json
import os
import subprocess
import sys

import pytest

E2E = "C:/Users/hm/Projects/e2e"
MODEL = "tiny-moe"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(E2E, MODEL)),
    reason="e2e fixture absent",
)


def _train_env(tmp_path, extra):
    """Run the real CLI on a fixture, with extra flags appended."""
    cmd = [
        sys.executable, "-m", "usaf.train",
        "--model", os.path.join(E2E, MODEL),
        "--dataset", os.path.join(E2E, "data.jsonl"),
        "--seq-len", "32",
        "--steps", "2",
        "--eval-every", "0",
        "--save-every", "0",
        "--checkpoint-dir", str(tmp_path),
        "--log-dir", str(tmp_path / "logs"),
    ]
    cmd += extra
    r = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=1800,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_a_successful_export_still_works(tmp_path):
    """The success path must keep working; making failure fatal is not enough."""
    out_path = tmp_path / "merged"
    rc, out = _train_env(tmp_path, ["--export", str(out_path), "--frac", "0.1"])
    assert rc == 0, out[-2000:]
    assert "Exported:" in out, out[-2000:]
    assert out_path.exists(), "nothing was exported"


def test_a_failing_export_is_not_reported_as_success(tmp_path):
    """The export must be fatal, not a line in the log."""
    # The parent of the output path is a regular file, so creating the output
    # directory cannot succeed. A file at the output path itself was tolerated -
    # the merge made the directory anyway - so this placement is the one that
    # actually exercises the failure path.
    parent = tmp_path / "occupied"
    parent.write_text("not a directory", encoding="utf-8")
    rc, out = _train_env(tmp_path, ["--export", str(parent / "model"), "--frac", "0.1"])
    assert rc != 0, (
        "a failed --export returned 0, so the run looked successful:\n"
        + out[-2000:]
    )
    assert "export was requested" in out, out[-2000:]


def test_an_unbuildable_foreign_model_stops(tmp_path):
    """A foreign architecture must not be silently swapped for Qwen3-MoE."""
    fake = tmp_path / "not-a-real-model"
    fake.mkdir(parents=True, exist_ok=True)
    (fake / "config.json").write_text(
        json.dumps({
            "model_type": "definitelyNotAMoe",
            "architectures": ["DefinitelyNotAMoeForCausalLM"],
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "vocab_size": 128,
        }),
        encoding="utf-8",
    )
    r = subprocess.run(
        [
            sys.executable, "-m", "usaf.train",
            "--model", str(fake),
            "--dataset", os.path.join(E2E, "data.jsonl"),
            "--seq-len", "16",
            "--steps", "1",
            "--checkpoint-dir", str(tmp_path / "ck"),
            "--log-dir", str(tmp_path / "logs"),
        ],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900,
    )
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode != 0, (
        "an unbuildable foreign model was not stopped:\n" + out[-2000:]
    )
