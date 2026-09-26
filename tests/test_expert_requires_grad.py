"""The weights of an expert must ask for gradients the same way every time.

get_expert_weights had three ways to answer and two different answers. A cache
miss returns nn.Parameter(t, requires_grad=True). A cache hit returns
nn.Parameter(t, requires_grad=t.requires_grad) - and what the cache holds is the
plain tensor _to_device produced, whose requires_grad is False. So the same
expert, the same weights, asked for gradients on the first lookup and refused
them on the second.

Nothing raises when that happens. It decides whether the post-accumulate hook
that feeds the sparse gradient store ever fires, so a run would capture
gradients from whichever experts happened to be a cache miss and silently
skip the rest - and whether an expert is a hit depends on max_cached and on how
far apart the two lookups are. The loss would still fall, because the layers
that did get gradients are real layers.
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


def test_two_lookups_of_the_same_expert_agree_about_needing_gradients(q_dict):
    cache = QuantizedExpertCache(q_dict, torch.device("cpu"), max_cached=2)
    mod = MODULES[0]
    first = cache.get_expert_weights(mod)
    second = cache.get_expert_weights(mod)
    assert mod in cache.cached_experts, "the second lookup was not a cache hit"
    want = {k: v.requires_grad for k, v in first.items()}
    got = {k: v.requires_grad for k, v in second.items()}
    assert got == want, f"{mod}: first {want}, second {got}"


def test_a_look_the_expert_still_needs_gradients(q_dict):
    cache = QuantizedExpertCache(q_dict, torch.device("cpu"), max_cached=2)
    for mod in MODULES:
        for pname, p in cache.get_expert_weights(mod).items():
            assert p.requires_grad, f"{mod}.{pname} does not require grad"


def test_it_survives_the_eviction_that_the_training_loop_does_every_step(q_dict):
    # train.py calls evict_all() after every forward, so in a run the answer
    # alternates hit, miss, hit, miss. A gradient store fed by that alternation
    # captures one step out of two and looks like it is learning.
    cache = QuantizedExpertCache(q_dict, torch.device("cpu"), max_cached=2)
    seen = []
    for _ in range(3):
        for mod in MODULES:
            seen.append(all(p.requires_grad
                            for p in cache.get_expert_weights(mod).values()))
        cache.evict_all()
    assert all(seen), (
        f"{seen.count(False)} of {len(seen)} lookups did not require gradients")


def test_and_after_a_prefetch_rather_than_a_plain_dequant(q_dict):
    # The prefetched branch is a third way of answering; it has to agree too.
    cache = QuantizedExpertCache(q_dict, torch.device("cpu"), max_cached=2)
    for mod in MODULES:
        cache.get_expert_weights(mod)
    cache.evict_all()
    for mod in MODULES:
        cache.prefetch(mod)
    for mod in MODULES:
        got = cache.get_expert_weights(mod)
        assert all(p.requires_grad for p in got.values()), mod
