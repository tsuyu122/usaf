"""pad_token_id 0 is a pad id, not a missing one.

train.py filled generate()'s pad_token_id with `tokenizer.pad_token_id or
tokenizer.eos_token_id`. When the pad id is 0 - ordinary, and used by plenty of
tokenizers - the `or` treated it as absent and filled with eos instead, which is
the one token the model has certainly been trained never to emit at the end of a
sequence. The run this matters for ends in a demonstration, and a demonstration
is exactly where a wrong pad token shows up.

So the value is checked on the way through, by watching what generate() is
actually called with.
"""
import contextlib
import io
import os

import pytest

E2E = r"C:\Users\hm\Projects\e2e"
MODEL = os.path.join(E2E, "tiny-moe")
Q4 = os.path.join(E2E, "tiny-moe-q4", "experts_q4.pt")
DATA = os.path.join(E2E, "data.jsonl")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(MODEL) and os.path.exists(Q4) and os.path.exists(DATA)),
    reason="e2e fixture absent",
)


class _Tok0:
    """pad id 0, eos 7, so the two answers are distinguishable."""

    pad_token_id = 0
    eos_token_id = 7

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __call__(self, *a, **kw):
        return self._inner(*a, **kw)


def _train_and_capture(mp, tmp_path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from usaf.train import main

    # AutoModelForCausalLM is a factory in transformers 5, and neither it nor
    # PreTrainedModel carries generate, so the class that actually defines it
    # is resolved from the model config and patched there.
    cls = type(AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL)))
    owner = next(k for k in cls.__mro__ if "generate" in k.__dict__)
    seen = {}
    real_gen = owner.__dict__["generate"]

    def spy(self, *a, **kw):
        seen.update(kw)
        return real_gen(self, *a, **kw)

    mp.setattr(owner, "generate", spy)
    tok = AutoTokenizer.from_pretrained(MODEL)
    mp.setattr("usaf.train._get_tokenizer", lambda *a, **k: _Tok0(tok))

    out = tmp_path / "o"
    out.mkdir(exist_ok=True)
    argv = [
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
        "--generate-after", "ola",
        "--generate-tokens", "4",
        "--checkpoint-dir", str(out / "ck"),
        "--log-dir", str(out / "logs"),
        "--tag", "pad",
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            main(argv)
        except SystemExit as e:
            if e.code not in (0, None):
                raise
    return seen, buf.getvalue()


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    """pytest monkeypatch is function-scoped and this is module-scoped, so the
    run happens once here and the assertions read what it recorded."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    try:
        yield _train_and_capture(mp, tmp_path_factory.mktemp("pad"))
    finally:
        mp.undo()


def test_generation_actually_happened(captured):
    seen, out = captured
    if not seen:
        raise AssertionError(
            "generate() was never reached; the run said:\n" + out[-3000:])
    assert "PROMPT: ola" in out, out[-1500:]


def test_a_pad_id_of_zero_is_passed_through_as_zero(captured):
    """The whole point. eos here is 7, so a 7 in this slot is the bug."""
    seen, _out = captured
    assert seen["pad_token_id"] == 0, seen.get("pad_token_id")
