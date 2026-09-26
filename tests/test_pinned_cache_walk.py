"""The pinned host cache, through the path a run actually uses.

A run walks layer after layer, asking for each expert in turn. Pinning every
one of them and keeping all of them means the host holds a second complete copy
of the quantized weights for the whole run, in memory that cannot be paged out
and cannot be reclaimed under pressure.
"""
import os

import pytest
import torch

from usaf.moe_loader import QuantizedExpertCache

E2E = r"C:\Users\hm\Projects\e2e"
Q4 = os.path.join(E2E, "tiny-moe-q4", "experts_q4.pt")

pytestmark = pytest.mark.skipif(not os.path.exists(Q4), reason="e2e fixture absent")


@pytest.fixture(scope="module")
def q_dict():
    return torch.load(Q4, map_location="cpu", weights_only=False)


def _modules(q_dict):
    names = set()
    for full in q_dict:
        mod, _ = full.rsplit(".", 1)
        names.add(mod)
    return sorted(names)


def test_walking_every_expert_does_not_pin_all_of_them(q_dict):
    cache = QuantizedExpertCache(q_dict, torch.device("cpu"), max_cached=1)
    for mod in _modules(q_dict):
        cache.get_expert_weights(mod)
        cache.evict_all()
    msg = f"{len(cache._pinned)} pinned entries for a bound of {cache._max_pinned}"
    assert len(cache._pinned) <= cache._max_pinned, msg


def test_a_walk_still_returns_the_right_weights(q_dict):
    # Bounding the cache must not change what it answers: the bound evicts
    # whatever it pinned first, so a later expert is read from the quantized
    # payload again and has to come back identical.
    from usaf.quantization import dequantize_4bit

    cache = QuantizedExpertCache(q_dict, torch.device("cpu"), max_cached=1)
    for mod in _modules(q_dict):
        got = cache.get_expert_weights(mod)
        for full, entry in q_dict.items():
            m, p = full.rsplit(".", 1)
            if m != mod:
                continue
            if isinstance(entry, dict):
                want = dequantize_4bit(entry["q"], entry["s"], entry["z"],
                                       entry["shape"], group_size=128)
            else:
                want = dequantize_4bit(entry[0], entry[1], entry[2],
                                       entry[3], group_size=128)
            have = got[p].detach().reshape(-1).float()
            assert torch.equal(have, want.reshape(-1).float()), full
            break
        cache.evict_all()

