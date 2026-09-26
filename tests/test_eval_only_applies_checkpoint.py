"""--eval-only has to evaluate the weights it was pointed at.

It used to run the benchmark and return before the resume block ever ran, so
the overlays were never applied. A model trained from loss 5.32 down to 1.37
benchmarked to exactly the same perplexity as the untouched base - 122.52,
identical to two decimals - and wrote that into the report without a word. The
failure is invisible by construction: the two answers looked the same, which is
what "the flag does nothing" always looks like.
"""
import os

import pytest
import torch

from usaf.moe_loader import QuantizedExpertCache
from usaf.train import _apply_resume_overlays

E2E = r"C:\Users\hm\Projects\e2e"
Q4 = os.path.join(E2E, "tiny-moe-q4", "experts_q4.pt")

pytestmark = pytest.mark.skipif(not os.path.exists(Q4), reason="e2e fixture absent")

MOD = "model.layers.0.mlp.experts"


@pytest.fixture(scope="module")
def q_dict():
    return torch.load(Q4, map_location="cpu", weights_only=False)


def _checkpoint(shift):
    n = 32
    idx = torch.arange(n, dtype=torch.long) * 7
    master = torch.full((n,), float(shift))
    return {
        "step": 40,
        "active_idx": {MOD + ".gate_up_proj": idx},
        "masters": {MOD + ".gate_up_proj": master},
    }


def test_overlays_are_registered_on_the_cache(q_dict):
    dev = torch.device("cpu")
    cache = QuantizedExpertCache(q_dict, dev, max_cached=1)
    idx = _apply_resume_overlays(_checkpoint(3.0), cache)
    name = MOD + ".gate_up_proj"
    assert name in cache.overlays
    got_idx, got_vals = cache.overlays[name]
    assert torch.equal(got_idx, torch.arange(32, dtype=torch.long) * 7)
    assert torch.allclose(got_vals.detach(), torch.full((32,), 3.0))
    assert torch.equal(idx[name], got_idx)


def test_the_benchmarked_weights_actually_change(q_dict):
    """Not just that the overlay was registered: that it reaches the weights
    the model is served. Two different checkpoints have to give two different
    tensors, or the benchmark is reading the base model either way.
    """
    dev = torch.device("cpu")

    def served(shift):
        c = QuantizedExpertCache(q_dict, dev, max_cached=1)
        _apply_resume_overlays(_checkpoint(shift), c)
        w = c.get_expert_weights(MOD)["gate_up_proj"]
        return w.detach().reshape(-1).float()

    base = QuantizedExpertCache(q_dict, dev, max_cached=1)
    plain = base.get_expert_weights(MOD)["gate_up_proj"].detach().reshape(-1).float()

    a = served(3.0)
    b = served(9.0)

    assert not torch.allclose(a, plain), "the overlay did not reach the weights"
    assert not torch.allclose(a, b), "two checkpoints served the same tensor"
