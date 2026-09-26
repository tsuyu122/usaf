import torch

import usaf.moe_loader as ml_loader  # noqa: N812
from usaf.moe_loader import QuantizedExpertCache


def _cache(max_cached=2):
    return QuantizedExpertCache({}, torch.device("cpu"), max_cached=max_cached)


def _params(tag, n=3):
    return {f"w{i}": torch.full((4, 4), float(tag)) for i in range(n)}


def test_the_pinned_host_cache_is_bounded_across_a_whole_model_walk():
    c = _cache(max_cached=2)
    for layer in range(60):
        c._to_device(f"model.layers.{layer}.mlp.experts", _params(layer))
    assert len(c._pinned) <= c._max_pinned, len(c._pinned)


def test_mutation_unbounded_pinned_cache_is_caught(monkeypatch):
    # The same walk against a cache whose bound was removed. If this passes,
    # the assertion above is not pinning anything.
    monkeypatch.setattr(ml_loader.QuantizedExpertCache, "_max_pinned", 10**9,
                        raising=False)
    c = _cache(max_cached=2)
    c._max_pinned = 10**9
    for layer in range(60):
        c._to_device(f"model.layers.{layer}.mlp.experts", _params(layer))
    assert len(c._pinned) > 16, len(c._pinned)

