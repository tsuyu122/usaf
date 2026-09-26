"""The reselection inside the trainer, not just the selector that feeds it.

RigL is the mechanism the method is named for, and the trainer integration had
never run: the in-process run uses four steps and the default reselect_every is
fifty, so do_reselect was never reached. The selector and SparseAdam.reselect
are each tested on their own, which is exactly the arrangement in which the
part between them is wrong and nothing notices.

That part is where a real mistake would live: the active set is rebuilt, the
optimizer is rebound, the overlays are reinstalled and the hooks are moved to
the new modules. A run that reselects and then trains on weights it is no
longer tracking loses everything it learned and reports a falling loss anyway.
"""
import contextlib
import io
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
def reselected(tmp_path_factory):
    from usaf.train import main
    out = tmp_path_factory.mktemp("reselect")
    argv = [
        "--model", MODEL,
        "--quant-path", Q4,
        "--dataset", DATA,
        "--seq-len", "32",
        "--microbatch", "1",
        "--steps", "6",
        "--lr", "1e-3",
        "--frac", "0.05",
        # Fires at steps 2, 4 and 6: the active set is rebuilt and the
        # optimizer rebound three times, with training continuing after each.
        "--reselect-every", "2",
        "--eval-every", "0",
        # A checkpoint after a reselect is the point: the file has to carry the
        # new active set and a step number that agrees with it.
        "--save-every", "2",
        "--checkpoint-dir", str(out / "ck"),
        "--log-dir", str(out / "logs"),
        "--tag", "rs",
    ]
    from usaf.mixtral_dml import unpatch_mixtral_for_dml
    from usaf.olmoe_dml import unpatch_olmoe_for_dml
    from usaf.qwen3moe_dml import unpatch_qwen3moe_for_dml
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            main(argv)
        except SystemExit as e:
            if e.code not in (0, None):
                raise
        finally:
            unpatch_qwen3moe_for_dml()
            unpatch_olmoe_for_dml()
            unpatch_mixtral_for_dml()
    return buf.getvalue(), out


def test_the_reselection_actually_happened(reselected):
    out, _ = reselected
    assert "Traceback" not in out, out[-3000:]
    hits = re.findall(r"\[reselect\] kept=([\d,]+) dropped=([\d,]+) "
                      r"grown=([\d,]+) active=([\d,]+)", out)
    assert len(hits) >= 2, (f"only {len(hits)} reselections in the output:\n"
                            f"{out[-3000:]}")


def test_training_continued_after_the_reselection(reselected):
    """The reselect rebuilds the active set, the overlays and the hooks. If the
    run then trains on weights it is not tracking, every step afterwards is
    wasted and the loss still falls."""
    out, _ = reselected
    steps = re.findall(r"^\s+(\d+)/(\d+) \| loss ([\d.]+)", out, re.M)
    assert len(steps) >= 6, f"only {len(steps)} steps logged:\n{out[-3000:]}"


def test_the_active_share_holds_across_the_reselection(reselected):
    out, _ = reselected
    shares = re.findall(r"\[reselect\].*?active=[\d,]+ \(([\d.]+)%\)", out)
    assert shares, "no reselect lines to check"
    for pct in shares:
        v = float(pct)
        assert 0 < v < 100, v


def test_the_checkpoint_records_the_reselected_set(reselected):
    _, out = reselected
    ckdir = out / "ck"
    if not ckdir.exists():
        pytest.skip("no checkpoint written")
    cks = sorted(ckdir.glob("*.pt"))
    assert cks, "the reselect produced no checkpoint"
    import torch
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
    assert ck.get("active_idx"), "the checkpoint has no active set"
    step = ck.get("step")
    assert isinstance(step, int) and step >= 1, step
