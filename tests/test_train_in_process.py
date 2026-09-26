"""The trainer, run in this process rather than in a subprocess.

Every end-to-end test here shells out, which is the right way to test a command
line but the wrong way to find out what it executes: a subprocess contributes
nothing to coverage, and train.py sat at 20% while passing every test anyone
had written. main() takes its arguments directly, so the same run happens here
and the lines it actually walks are counted.

The properties asserted are the ones a broken run still looks healthy without:
a loss that falls, an active set that is neither empty nor the whole weight,
and a log file that exists afterwards.
"""
import contextlib
import io
import json
import os
import re

import pytest

E2E = r"C:\Users\hm\Projects\e2e"
MODEL = os.path.join(E2E, "tiny-moe")
Q4 = os.path.join(E2E, "tiny-moe-q4", "experts_q4.pt")
DATA = os.path.join(E2E, "data.jsonl")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(MODEL) and os.path.exists(Q4) and os.path.exists(DATA)),
    reason="e2e fixture absent",
)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from usaf.train import main
    out = tmp_path_factory.mktemp("inproc")
    argv = [
        "--model", MODEL,
        "--quant-path", Q4,
        "--dataset", DATA,
        "--seq-len", "32",
        "--microbatch", "1",
        "--steps", "4",
        "--lr", "1e-3",
        "--frac", "0.05",
        "--eval-every", "0",
        "--save-every", "0",
        "--checkpoint-dir", str(out / "ck"),
        "--log-dir", str(out / "logs"),
        "--tag", "ip",
    ]
    # capsys is function-scoped and this fixture is module-scoped, so the run
    # happens exactly once and the assertions read what it printed. redirect
    # catches the prints, which is all that matters here.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            main(argv)
        except SystemExit as e:
            if e.code not in (0, None):
                raise
    return buf.getvalue(), out


def test_the_run_completes_inside_this_process(run):
    out, _ = run
    assert "Traceback" not in out, out[-3000:]


def test_a_loss_was_recorded_for_every_step(run):
    out, _ = run
    # The training lines are "    1/4 | loss 5.1055 | 103 tok/s | ...". The
    # importance lines below them are "  imp 1/3 | loss ..." and are a
    # different pass over the same data, so matching on the wrong one of the
    # two would pass while the training loop itself logged nothing.
    steps = re.findall(r"^\s+(\d+)/(\d+) \| loss ([\d.]+)", out, re.M)
    assert len(steps) >= 4, f"only {len(steps)} steps logged:\n{out[-3000:]}"
    for n, _, v in steps:
        assert v not in ("nan", "inf"), f"step {n} loss {v}"


def test_the_loss_goes_down(run):
    """A run that logs a plausible number every step can still be a run that
    learned nothing - the numbers come from the forward pass, which is the
    part that always works. Only the trend says the sparse update reached the
    weights."""
    out, _ = run
    steps = re.findall(r"^\s+(\d+)/(\d+) \| loss ([\d.]+)", out, re.M)
    losses = [float(v) for _, _, v in steps]
    assert losses[-1] < losses[0], losses


def test_the_active_set_is_a_small_non_empty_share(run):
    out, _ = run
    m = re.search(r"Active:\s*([\d,]+)/([\d,]+)", out)
    assert m, "no Active line - selection never happened:\n" + out[-3000:]
    active = int(m.group(1).replace(",", ""))
    total = int(m.group(2).replace(",", ""))
    assert 0 < active < total, f"{active} of {total}"


def test_the_log_file_written_during_the_run_is_readable(run):
    _, out = run
    logdir = out / "logs"
    logs = list(logdir.glob("*.jsonl"))
    assert logs, "nothing written to the log directory"
    text = logs[0].read_text(encoding="utf-8")
    rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    assert rows, "the log file is empty"
    for row in rows:
        assert "loss" in row, row


def test_the_trainer_reports_the_backend_it_used(run):
    """Which device ran is the first thing to be wrong and the easiest to lose."""
    out, _ = run
    m = re.search(r"Backend:\s*([^\n]+)", out)
    assert m, "no Backend line:\n" + out[-3000:]
    label = m.group(1).strip()
    # The label has to name a backend that exists. It used to be
    # "DirectML/CPU" on every non-CUDA run, including the ones where
    # importing DirectML failed and the device was the CPU, so the log claimed
    # a run that did not happen. A slash-separated pair of names is not a
    # backend and is not readable as either one.
    assert label.split()[0] in ("CUDA", "DirectML", "CPU"), label
