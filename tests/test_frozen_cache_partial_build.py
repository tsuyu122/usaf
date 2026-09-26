import numpy as np
import pytest
import torch

from usaf.frozen_cache import build_frozen_cache, load_frozen_cache


def _samples(n=4, seq=8):
    return [{"input_ids": np.arange(i * seq, (i + 1) * seq, dtype=np.int64)}
            for i in range(n)]


def _hidden(sample):
    return torch.ones(8, 16, dtype=torch.float16)


def test_a_build_that_dies_midway_is_not_accepted(tmp_path):
    # A build writes the array before the metadata, and the w+ open truncates
    # the array on the way in. A crash between them leaves the previous run's
    # metadata describing an array that is no longer there, and every check
    # still passes. The tail of the cache is then training data.
    path = str(tmp_path / "fc.npy")
    build_frozen_cache(_samples(), 8, 16, 2, "src", _hidden, path, verbose=False)
    assert load_frozen_cache(_samples(), 8, 16, 2, "src", path) is not None

    def _explode(sample):
        raise RuntimeError("machine died mid-build")

    with pytest.raises(RuntimeError):
        # A different src changes the fingerprint, so this is a real rebuild
        # and the callback is actually called.
        build_frozen_cache(_samples(), 8, 16, 2, "other", _explode, path,
                          verbose=False)

    # A crash must leave nothing that looks like a finished cache.
    assert load_frozen_cache(_samples(), 8, 16, 2, "src", path) is None

