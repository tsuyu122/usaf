"""The evaluator, and the two ways it used to fail on a run that went wrong.

A diverged run is exactly when a person most wants to see the perplexity, and
math.exp raises OverflowError past about 709 - so the number that explains the
divergence is the one thing the evaluation could not print. And generate_sample
returned the whole decoded sequence, prompt included, so a caller printing the
prompt and then the result showed the sentence twice and made it look like the
model had echoed it.
"""
import math

import pytest
import torch
import torch.nn as nn

from usaf.evaluate import Evaluator


class _Batch(dict):
    """Stands in for BatchEncoding, which is a dict that also has .to()."""

    def to(self, device):
        return self


class _Tok:
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, **kw):
        return _Batch(input_ids=torch.tensor([[1, 2, 3]]),
                      attention_mask=torch.ones(1, 3))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids.tolist())


class _Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None
        self.loss_value = 1.0

    def forward(self, input_ids=None, labels=None, **kw):
        loss = None
        if labels is not None:
            loss = torch.tensor(self.loss_value, requires_grad=True)
        return type("O", (), {"loss": loss})()

    def generate(self, input_ids=None, attention_mask=None, **kw):
        self.seen = kw
        return torch.cat([input_ids, torch.tensor([[7, 8, 9, 10]])], dim=-1)


def _loader(n=2, batch=2, seq=4):
    return [{"input_ids": torch.zeros(batch, seq, dtype=torch.long),
             "labels": torch.zeros(batch, seq, dtype=torch.long)}
            for _ in range(n)]


def test_generation_returns_only_what_came_after_the_prompt():
    m = _Gen()
    ev = Evaluator(m, torch.device("cpu"), tokenizer=_Tok())
    out = ev.generate_sample("anything", max_new_tokens=4, temperature=0.7)
    assert out == "7 8 9 10", out


def test_generation_can_be_greedy():
    # temperature 0 used to be passed through with do_sample=True, which most
    # backends reject or silently ignore. It now means greedy, which is what
    # anyone comparing two runs of the same prompt actually wants.
    m = _Gen()
    ev = Evaluator(m, torch.device("cpu"), tokenizer=_Tok())
    ev.generate_sample("anything", max_new_tokens=4, temperature=0.0)
    assert m.seen["do_sample"] is False
    assert "temperature" not in m.seen


def test_perplexity_is_reported_for_a_normal_loss():
    m = _Gen()
    ev = Evaluator(m, torch.device("cpu"), tokenizer=_Tok())
    r = ev.evaluate_perplexity(_loader(n=2))
    assert r["loss"] == pytest.approx(1.0)
    assert r["perplexity"] == pytest.approx(math.e)
    assert r["total_tokens"] == 16


def test_a_diverged_run_reports_instead_of_raising():
    m = _Gen()
    m.loss_value = 5000.0
    ev = Evaluator(m, torch.device("cpu"), tokenizer=_Tok())
    r = ev.evaluate_perplexity(_loader(n=1))
    assert r["loss"] == pytest.approx(5000.0)
    assert math.isinf(r["perplexity"]), r


def test_generation_without_a_tokenizer_says_so():
    ev = Evaluator(_Gen(), torch.device("cpu"), tokenizer=None)
    with pytest.raises(ValueError, match="tokenizer is required"):
        ev.generate_sample("anything")

