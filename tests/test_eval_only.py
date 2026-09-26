"""Regression tests for --eval-only in the universal CLI.

--eval-only was completely non-functional: it passed tokenizer=None into
run_benchmark because the model is built with from_config and therefore has no
tokenizer attribute, and the global fallback was only ever fed from that same
non-existent attribute. The failure surfaced a dozen frames deep as a
NoneType-not-callable TypeError from inside perplexity.py.
"""
import pytest
import torch

from usaf.eval.perplexity import compute_perplexity


class _Tok:
    'Minimal tokenizer stub: every id is valid for the stub model.'

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    vocab_size = 32

    def __len__(self):
        return 32

    def __call__(self, text, **kw):
        ids = [1 + (i % 30) for i, _ in enumerate(text.split())]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _Out:
    def __init__(self, loss):
        self.loss = loss


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab = 32

    def forward(self, input_ids=None, labels=None, **kw):
        return _Out(torch.tensor(1.5))


def test_compute_perplexity_rejects_none_tokenizer():
    'A None tokenizer must fail with a message that names the cause.'
    with pytest.raises(ValueError, match="needs a tokenizer"):
        compute_perplexity(_Model(), None, ["int main"], torch.device("cpu"))


def test_compute_perplexity_rejects_non_callable_tokenizer():
    with pytest.raises(TypeError, match="callable tokenizer"):
        compute_perplexity(_Model(), "not-a-tokenizer", ["x"], torch.device("cpu"))


def test_compute_perplexity_works_with_a_real_tokenizer():
    out = compute_perplexity(_Model(), _Tok(), ["int main", "return 0"],
                             torch.device("cpu"), verbose=False)
    assert out["tokens"] > 0
    assert out["perplexity"] == pytest.approx(4.4817, rel=1e-3)


def test_compute_perplexity_raises_instead_of_reporting_perfect_ppl():
    """Zero scored tokens used to yield perplexity 1.0, which is
    indistinguishable from a perfect model in a results file."""

    class _Silent(_Model):
        def forward(self, input_ids=None, labels=None, **kw):
            return _Out(None)

    with pytest.raises(RuntimeError, match="no tokens were scored"):
        compute_perplexity(_Silent(), _Tok(), ["a", "b"], torch.device("cpu"),
                           verbose=False)
