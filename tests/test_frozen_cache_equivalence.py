"""The frozen cache must reproduce the forward it replaces, not approximate it.

--no-frozen-cache in the universal CLI used to be a flag that changed nothing:
the config field was read and printed, and no cache was ever built. Now that
the flag actually does something, the invariant it rests on needs pinning - the
cached hidden state has to be what the frozen prefix would have produced, or
training silently runs on different activations than the evaluation reports.
"""
import numpy as np
import pytest
import torch

from usaf.frozen_cache import build_frozen_cache, dataset_fingerprint, get_hidden


class _Prefix(torch.nn.Module):
    """A stand-in for layers 0..DETACH_AT."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lin = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, h):
        return torch.tanh(self.lin(h))


def _samples(n, seq, hidden, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        {
            "input_ids": torch.randint(0, 100, (seq,), generator=g).tolist(),
            "labels": torch.zeros(seq, dtype=torch.long).tolist(),
        }
        for _ in range(n)
    ]


def test_cached_hidden_matches_a_fresh_forward(tmp_path):
    """The whole premise: a frozen prefix gives the same answer every time."""
    torch.manual_seed(0)
    seq, hidden, n = 3, 4, 5
    prefix = _Prefix(hidden).double()
    samples = _samples(n, seq, hidden)

    def compute_hidden(s):
        assert len(s["input_ids"]) == seq
        h = torch.randn(seq, hidden, generator=torch.Generator().manual_seed(1)).double()
        return prefix(h).unsqueeze(0)

    path = str(tmp_path / "fc.npy")
    cache = build_frozen_cache(
        samples, seq, hidden, 0, "test-model", compute_hidden, path, verbose=False
    )

    for i, s in enumerate(samples):
        want = compute_hidden(s).reshape(seq, hidden).to(torch.float16)
        got = get_hidden(cache, i, "cpu")
        assert got.shape == (1, seq, hidden), got.shape
        assert torch.equal(got[0], want), f"sample {i} differs from a fresh forward"


def test_cache_is_reused_when_the_fingerprint_matches(tmp_path):
    seq, hidden, n = 2, 4, 3
    samples = _samples(n, seq, hidden)
    path = str(tmp_path / "fc.npy")

    calls = []

    def compute_hidden(s):
        calls.append(1)
        return torch.zeros(1, seq, hidden)

    build_frozen_cache(samples, seq, hidden, 0, "m", compute_hidden, path, verbose=False)
    assert len(calls) == n
    build_frozen_cache(samples, seq, hidden, 0, "m", compute_hidden, path, verbose=False)
    assert len(calls) == n, "a matching fingerprint must not recompute anything"


def test_cache_is_rebuilt_when_the_dataset_changes(tmp_path):
    """A different dataset must not silently reuse the old activations."""
    seq, hidden = 2, 4
    path = str(tmp_path / "fc.npy")
    calls = []

    def compute_hidden(s):
        calls.append(1)
        return torch.zeros(1, seq, hidden)

    build_frozen_cache(
        _samples(3, seq, hidden), seq, hidden, 0, "m", compute_hidden, path,
        verbose=False,
    )
    before = len(calls)
    build_frozen_cache(
        _samples(4, seq, hidden, seed=99), seq, hidden, 0, "m", compute_hidden, path,
        verbose=False,
    )
    assert len(calls) > before, "a changed dataset must invalidate the cache"


def test_cache_is_rebuilt_when_the_detach_point_moves(tmp_path):
    seq, hidden = 2, 4
    samples = _samples(3, seq, hidden)
    path = str(tmp_path / "fc.npy")
    calls = []

    def compute_hidden(s):
        calls.append(1)
        return torch.zeros(1, seq, hidden)

    build_frozen_cache(samples, seq, hidden, 0, "m", compute_hidden, path, verbose=False)
    before = len(calls)
    build_frozen_cache(samples, seq, hidden, 1, "m", compute_hidden, path, verbose=False)
    assert len(calls) > before


def test_fingerprint_tracks_ids_layer_and_source():
    a = _samples(2, 2, 2, seed=1)
    b = _samples(2, 2, 2, seed=2)
    base = dataset_fingerprint(a, 0, "m")
    assert base != dataset_fingerprint(b, 0, "m"), "different ids must differ"
    assert base != dataset_fingerprint(a, 1, "m"), "a different layer must differ"
    assert base != dataset_fingerprint(a, 0, "other"), "a different model must differ"
    assert base == dataset_fingerprint(a, 0, "m"), "the same input must be stable"


def test_out_of_range_index_names_the_sample(tmp_path):
    """The trainer assigns one global index across train/eval/held-out, so an
    evaluation sample is routinely absent from a train-only cache."""
    cache = np.zeros((2, 2, 2), dtype=np.float16)
    with pytest.raises(IndexError, match="never cached"):
        get_hidden(cache, 5, "cpu")
