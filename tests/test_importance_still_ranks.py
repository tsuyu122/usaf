"""The importance pass has to still produce a ranking after the fix.

The forward it runs was moved under torch.no_grad, because the graph it built
was never differentiated - each layer is replayed from its stored input and the
gradient threaded through by hand. That fix has a failure mode with no visible
symptom: if the replay ever stopped reaching the experts, the importance store
would simply accumulate nothing, select nothing, and the run would train on an
empty set of weights and report a plausible loss the whole way. So this runs
the trainer and checks the selection is non-empty and the right size.
"""
import os
import re
import subprocess
import sys

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
    out = tmp_path_factory.mktemp("importance")
    cmd = [
        sys.executable, "-u", "-m", "usaf.train",
        "--model", MODEL,
        "--quant-path", Q4,
        "--dataset", DATA,
        "--seq-len", "32",
        "--microbatch", "1",
        "--steps", "2",
        "--lr", "1e-3",
        "--frac", "0.05",
        "--eval-every", "0",
        "--save-every", "0",
        "--checkpoint-dir", str(out / "ck"),
        "--log-dir", str(out / "logs"),
        "--tag", "imp",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    return p.stdout + p.stderr


def test_the_run_completes(run):
    assert "Traceback" not in run, run[-2500:]


def test_the_importance_pass_selects_a_non_empty_set(run):
    m = re.search(r"Active:\s*([\d,]+)/([\d,]+)\s*\(([\d.]+)%\)", run)
    assert m, "no Active line - the importance pass never selected:\n" + run[-2500:]
    active = int(m.group(1).replace(",", ""))
    total = int(m.group(2).replace(",", ""))
    assert active > 0, "importance selected nothing: " + m.group(0)
    assert total > 0
    frac = active / total
    assert 0.03 < frac < 0.07, f"expected about 5%, got {frac:.4f}"


def test_the_importance_pass_reports_a_finite_loss(run):
    losses = re.findall(r"imp \d+/\d+ \| loss ([\d.]+)", run)
    assert losses, "the importance pass printed no loss:\n" + run[-2500:]
    for v in losses:
        f = float(v)
        assert f == f and f < 1e4, f"importance loss is not finite: {v}"

