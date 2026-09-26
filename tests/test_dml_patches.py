"""The DirectML expert loops must compute what transformers computes.

They were written against the pre-4.53 expert signature,
``forward(self, hidden_states, weights)``. transformers >= 4.53 unified every
MoE family onto ``forward(self, hidden_states, top_k_index, top_k_weights)``,
so the patched container was called with three positional arguments and
declared two: a TypeError on the first expert forward, in all three backends.
Nothing had run this path - torch-directml does not load against torch 2.13 -
so it shipped broken.

These run the real forward and the patched one on the same inputs and compare,
which catches both a wrong signature and a wrong routing mask. A signature
check alone would not catch a patch that accepted the arguments and then
weighted the experts wrongly.
"""
import types

import pytest
import torch

FAMILIES = [
    ("qwen3moe_dml", "dml_qwen3_experts_forward",
     "transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeExperts"),
    ("mixtral_dml", "dml_mixtral_experts_forward",
     "transformers.models.mixtral.modeling_mixtral", "MixtralExperts"),
    ("olmoe_dml", "dml_experts_forward",
     "transformers.models.olmoe.modeling_olmoe", "OlmoeExperts"),
]


def _make(cls, hidden, inter, n_experts):
    # Mixtral and Olmoe read num_local_experts, Qwen3-MoE reads num_experts;
    # supplying both keeps one fixture working for all three.
    cfg = types.SimpleNamespace(
        num_experts=n_experts,
        num_local_experts=n_experts,
        hidden_size=hidden,
        moe_intermediate_size=inter,
        intermediate_size=inter,
        hidden_act="silu",
        _experts_implementation="eager",
    )
    m = cls(cfg).to(torch.float32).eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m


def _routing(n_tokens, n_experts, top_k, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack(
        [torch.randperm(n_experts, generator=g)[:top_k] for _ in range(n_tokens)]
    )
    w = torch.softmax(torch.randn(n_tokens, top_k, generator=g), dim=-1)
    return idx, w


@pytest.mark.parametrize("module,fn,realmod,cls", FAMILIES)
def test_dml_patch_signature_matches_the_real_forward(module, fn, realmod, cls):
    import importlib
    import inspect

    patch = getattr(importlib.import_module("usaf." + module), fn)
    real = getattr(importlib.import_module(realmod), cls)
    assert list(inspect.signature(patch).parameters) == list(
        inspect.signature(real.forward).parameters
    )


@pytest.mark.parametrize("module,fn,realmod,cls", FAMILIES)
def test_dml_patch_computes_the_same_thing_as_transformers(module, fn, realmod, cls):
    import importlib

    real_cls = getattr(importlib.import_module(realmod), cls)
    patch = getattr(importlib.import_module("usaf." + module), fn)
    hidden, inter, n_experts, top_k, n_tokens = 32, 24, 4, 2, 7

    experts = _make(real_cls, hidden, inter, n_experts)
    hs = torch.randn(n_tokens, hidden)
    idx, w = _routing(n_tokens, n_experts, top_k)

    with torch.no_grad():
        expected = experts(hs, idx, w)

        original = real_cls.forward
        real_cls.forward = patch
        try:
            got = experts(hs, idx, w)
        finally:
            real_cls.forward = original

    rel = (expected - got.float()).abs().max() / expected.abs().max()
    assert rel < 1e-5, f"{module} differs from transformers by {rel:.2e} relative"
