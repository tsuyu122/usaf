"""Tests for the frozen activation cache.

The index bookkeeping here is the part that silently corrupted evaluation: the
trainer assigns one global _fidx across train, eval and held-out samples, so a
cache built from the training samples alone leaves every evaluation sample
holding an out-of-range index.
"""
import numpy as np
import pytest
import torch

from usaf.frozen_cache import dataset_fingerprint, get_hidden


def _cache(n=3, seq=2, hidden=2):
    arr = np.arange(n * seq * hidden, dtype=np.float16).reshape(n, seq, hidden)
    return arr


def test_get_hidden_returns_batch_dimension():
    t = get_hidden(_cache(), 1, "cpu")
    assert t.shape == (1, 2, 2)
    assert t.dtype == torch.float16


def test_get_hidden_out_of_range_raises():
    """An uncached sample must raise, not read another sample activations."""
    with pytest.raises(IndexError, match="out of range"):
        get_hidden(_cache(n=3), 3, "cpu")


def test_get_hidden_does_not_wrap_silently():
    """Index 3 into a 3-entry cache must fail, not quietly return entry 0."""
    cache = _cache(n=3)
    with pytest.raises(IndexError):
        get_hidden(cache, len(cache), "cpu")


def test_get_hidden_supports_negative_index():
    assert float(get_hidden(_cache(), -1, "cpu").flatten()[0]) == 8.0


def test_fingerprint_changes_with_content():
    a = [{"input_ids": [1, 2, 3]}]
    b = [{"input_ids": [1, 2, 4]}]
    assert dataset_fingerprint(a, 0, "m") != dataset_fingerprint(b, 0, "m")


def test_fingerprint_changes_with_detach_at_and_source():
    a = [{"input_ids": [1, 2, 3]}]
    base = dataset_fingerprint(a, 0, "m")
    assert dataset_fingerprint(a, 1, "m") != base
    assert dataset_fingerprint(a, 0, "other") != base


def test_fingerprint_is_stable():
    a = [{"input_ids": [1, 2, 3]}]
    assert dataset_fingerprint(a, 0, "m") == dataset_fingerprint(a, 0, "m")
