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

def _merge_like_eval_cli(param, aidx, trained):
    """Mirror the merge in usaf/eval_cli.py::main exactly.

    Used to pin the two failure modes seen there: copy_() of a 1-D scatter
    result into a multi-dimensional parameter, and an in-place scatter into a
    reshape that silently copied because the parameter was not contiguous.
    """
    with torch.no_grad():
        flat = param.data.reshape(-1)
        if flat.data_ptr() != param.data.data_ptr():
            param.data = flat.scatter(0, aidx, trained).reshape(param.shape)
        else:
            flat.scatter_(0, aidx, trained)


def test_eval_cli_merge_writes_into_a_multidimensional_parameter():
    p = torch.nn.Parameter(torch.zeros(4, 8, 16))
    aidx = torch.tensor([0, 100, 511])
    trained = torch.tensor([5.0, 6.0, 7.0])
    _merge_like_eval_cli(p, aidx, trained)
    flat = p.data.reshape(-1)
    assert float(flat[0]) == 5.0
    assert float(flat[100]) == 6.0
    assert float(flat[511]) == 7.0
    assert p.shape == (4, 8, 16), "the parameter must keep its original shape"


def test_eval_cli_merge_survives_a_non_contiguous_parameter():
    """A transposed parameter makes reshape(-1) return a copy, so an in-place
    scatter would be written to a throwaway tensor and silently lost.
    """
    base = torch.zeros(8, 4)
    p = torch.nn.Parameter(base.t())  # non-contiguous view
    assert not p.data.is_contiguous()
    aidx = torch.tensor([1, 20])

    _merge_like_eval_cli(p, aidx, torch.tensor([3.0, 4.0]))
    flat = p.data.reshape(-1)
    assert float(flat[1]) == 3.0
    assert float(flat[20]) == 4.0


def test_eval_cli_merge_leaves_untouched_positions_alone():
    p = torch.nn.Parameter(torch.arange(32, dtype=torch.float32))
    before = p.data.clone()
    _merge_like_eval_cli(p, torch.tensor([5, 9]), torch.tensor([100.0, 200.0]))
    # The positions already held 5 and 9, so the deltas are 100-5 and 200-9.
    assert float(p.data[5]) == 100.0
    assert float(p.data[9]) == 200.0
    diff = (p.data - before).abs()
    assert int((diff > 0).sum()) == 2, "only the active positions may change"
