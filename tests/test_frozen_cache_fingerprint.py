"""The frozen cache decides whether to reuse activations from a previous run.

Getting that decision wrong is silent: every shape check passes, the run
starts, and it trains on activations belonging to different samples. The
fingerprint is the only thing standing between a stale cache and that, and it
had a collision.
"""

import numpy as np
import pytest
import torch

from usaf.frozen_cache import build_frozen_cache, dataset_fingerprint, get_hidden, load_frozen_cache


@pytest.fixture
def samples():
    return [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5, 6, 7, 8]}]


@pytest.fixture
def cache_file(tmp_path):
    return str(tmp_path / "fc.npy")


def _build(samples, path, seq=4, hidden=3, detach_at=2, src="tiny-moe"):
    counter = []

    def compute_hidden(s):
        val = float(len(counter) + 1)
        counter.append(val)
        return torch.full((seq, hidden), val)

    return build_frozen_cache(samples, seq, hidden, detach_at, src,
                             compute_hidden, path, verbose=False)


def test_build_then_load_returns_the_same_values(samples, cache_file):
    arr = _build(samples, cache_file)
    again = load_frozen_cache(samples, 4, 3, 2, "tiny-moe", cache_file)
    assert again is not None
    assert np.array_equal(np.asarray(arr), np.asarray(again))


def test_fingerprint_distinguishes_how_tokens_split_between_samples():
    """[1,2],[3] and [1],[2,3] hold the same tokens in the same order but
    describe different samples, and the cache rows are positional. Concatenating
    them with no separator gave both the same digest.
    """
    a = [{"input_ids": [1, 2]}, {"input_ids": [3]}]
    b = [{"input_ids": [1]}, {"input_ids": [2, 3]}]
    assert dataset_fingerprint(a, 4, "src") != dataset_fingerprint(b, 4, "src")


def test_a_different_detach_layer_invalidates(samples, cache_file):
    _build(samples, cache_file)
    assert load_frozen_cache(samples, 4, 3, 3, "tiny-moe", cache_file) is None


def test_a_different_model_source_invalidates(samples, cache_file):
    _build(samples, cache_file)
    assert load_frozen_cache(samples, 4, 3, 2, "outro", cache_file) is None


def test_a_different_seq_or_hidden_invalidates(samples, cache_file):
    _build(samples, cache_file)
    assert load_frozen_cache(samples, 8, 3, 2, "tiny-moe", cache_file) is None
    assert load_frozen_cache(samples, 4, 5, 2, "tiny-moe", cache_file) is None


def test_different_samples_invalidate_the_cache(cache_file):
    first = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5, 6, 7, 8]}]
    _build(first, cache_file)
    other = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [9, 9, 9, 9]}]
    assert load_frozen_cache(other, 4, 3, 2, "tiny-moe", cache_file) is None


def test_a_different_sample_count_invalidates(samples, cache_file):
    _build(samples, cache_file)
    longer = samples + [{"input_ids": [1, 1, 1, 1]}]
    assert load_frozen_cache(longer, 4, 3, 2, "tiny-moe", cache_file) is None


def test_unreadable_metadata_invalidates_instead_of_crashing(samples, cache_file):
    _build(samples, cache_file)
    with open(cache_file + ".json", "w") as fh:
        fh.write("{nao e json")
    assert load_frozen_cache(samples, 4, 3, 2, "tiny-moe", cache_file) is None


def test_missing_cache_is_none_not_an_error(samples, tmp_path):
    missing = str(tmp_path / "nao-existe.npy")
    assert load_frozen_cache(samples, 4, 3, 2, "tiny-moe", missing) is None


def test_get_hidden_rejects_an_index_outside_the_cache(samples, cache_file):
    """A sample that was never cached must not be silently skipped."""
    arr = _build(samples, cache_file)
    with pytest.raises(IndexError):
        get_hidden(arr, 5, torch.device("cpu"))
    with pytest.raises(IndexError):
        get_hidden(arr, -5, torch.device("cpu"))


def test_get_hidden_accepts_negative_indices_from_the_end(samples, cache_file):
    arr = _build(samples, cache_file)
    last = get_hidden(arr, -1, torch.device("cpu"))
    first = get_hidden(arr, 0, torch.device("cpu"))
    assert not torch.equal(last, first)
