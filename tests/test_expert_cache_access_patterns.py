"""The expert cache has to survive an access pattern, not a single lookup.

max_cached=1 means every forward evicts the previous layer and dequantizes
the next one, and the resident path is supposed to answer from RAM instead. A
cache that returns a stale or half-updated entry would only show up partway
through a run - after a few layers, with a plausible loss, on the wrong
weights. So these walk an interleaved pattern and check every answer against
a dequantization made fresh each time.
"""
import os

import pytest
import torch

from usaf.moe_loader import QuantizedExpertCache

E2E = r"C:\Users\hm\Projects\e2e"
Q4 = os.path.join(E2E, "tiny-moe-q4", "experts_q4.pt")

pytestmark = pytest.mark.skipif(not os.path.exists(Q4), reason="e2e fixture absent")

MODULES = [f"model.layers.{i}.mlp.experts" for i in range(4)]


@pytest.fixture(scope="module")
def q_dict():
    return torch.load(Q4, map_location="cpu", weights_only=False)


def _reference(q_dict, mod, dev):
    """A dequantization that shares no state with the cache."""
    from usaf.quantization import dequantize_4bit

    out = {}
    for full in q_dict:
        m, p = full.rsplit(".", 1)
        if m != mod:
            continue
        e = q_dict[full]
        out[p] = dequantize_4bit(e[0], e[1], e[2], e[3], group_size=128)
    return {k: v.to(dev) for k, v in out.items()}


def test_every_lookup_matches_a_fresh_dequantization(q_dict):
    dev = torch.device("cpu")
    cache = QuantizedExpertCache(q_dict, dev, max_cached=1)
    for rep in range(3):
        for mod in MODULES:
            got = cache.get_expert_weights(mod)
            want = _reference(q_dict, mod, dev)
            assert set(got) == set(want), mod
            for p in want:
                d = float((got[p].detach().float() - want[p].float()).abs().max())
                assert d < 1e-3, mod + "." + p + " drifted by " + format(d, ".2e")


def test_interleaved_lookups_do_not_hand_back_the_wrong_layer(q_dict):
    """max_cached=1 with an alternating pattern is the streaming case."""
    dev = torch.device("cpu")
    cache = QuantizedExpertCache(q_dict, dev, max_cached=1)
    want = {m: _reference(q_dict, m, dev) for m in MODULES}
    pattern = [0, 3, 1, 0, 2, 3, 1, 2, 0, 1, 3, 2]
    for i in pattern:
        mod = MODULES[i]
        got = cache.get_expert_weights(mod)
        for p in want[mod]:
            d = float((got[p].detach().float() - want[mod][p].float()).abs().max())
            assert d < 1e-3, (
                "layer " + str(i) + " param " + p + " is not its own weights"
            )


def test_a_wider_cache_returns_the_same_thing_as_a_narrow_one(q_dict):
    dev = torch.device("cpu")
    narrow = QuantizedExpertCache(q_dict, dev, max_cached=1)
    wide = QuantizedExpertCache(q_dict, dev, max_cached=4)
    for mod in MODULES:
        a = narrow.get_expert_weights(mod)
        b = wide.get_expert_weights(mod)
        for p in a:
            d = float((a[p].detach().float() - b[p].detach().float()).abs().max())
            assert d < 1e-3, "cache width changed the answer for " + p


def test_resident_answers_match_the_non_resident_ones(q_dict):
    """make_resident is supposed to be a speed-up, not a different answer."""
    dev = torch.device("cpu")
    plain = QuantizedExpertCache(q_dict, dev, max_cached=1)
    resident = QuantizedExpertCache(q_dict, dev, max_cached=1)
    resident.make_resident(MODULES)
    for mod in MODULES:
        a = plain.get_expert_weights(mod)
        b = resident.get_expert_weights(mod)
        assert set(a) == set(b), mod
        for p in a:
            d = float((a[p].detach().float() - b[p].detach().float()).abs().max())
            assert d < 1e-3, "the resident path returned different weights for " + p


def test_partial_residency_fills_in_the_parameters_it_omitted(q_dict):
    """only_params is an optimization: the rest still has to be dequantized."""
    dev = torch.device("cpu")
    full = QuantizedExpertCache(q_dict, dev, max_cached=1)
    partial = QuantizedExpertCache(q_dict, dev, max_cached=1)
    partial.make_resident(MODULES, only_params=["gate_up_proj"])
    for mod in MODULES:
        a = full.get_expert_weights(mod)
        b = partial.get_expert_weights(mod)
        assert set(b) == set(a), "the omitted param came back empty"
        for p in a:
            d = float((a[p].detach().float() - b[p].detach().float()).abs().max())
            assert d < 1e-3, "partial residency changed the answer for " + p
