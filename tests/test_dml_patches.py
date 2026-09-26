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


@pytest.fixture(autouse=True)
def _leave_the_patches_off():
    """These tests swap a class attribute and put it back themselves, which is
    enough until a test fails partway: the process is then left with the DML
    forward installed and every later test runs it unknowingly. Cleaned up
    here rather than in the test body, so it holds when the body raises."""
    from usaf.mixtral_dml import unpatch_mixtral_for_dml
    from usaf.olmoe_dml import unpatch_olmoe_for_dml
    from usaf.qwen3moe_dml import unpatch_qwen3moe_for_dml
    yield
    unpatch_qwen3moe_for_dml()
    unpatch_olmoe_for_dml()
    unpatch_mixtral_for_dml()

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
def test_dml_patch_accepts_everything_the_real_forward_is_called_with(
        module, fn, realmod, cls):
    """The caller passes three positional arguments; the patch must take them.

    This used to compare the patch against whatever cls.forward happened to be
    at the time, which - because the trainer runs in-process and installs the
    patch globally - was the patch itself. The check was comparing the function
    with itself and could not fail.

    Equality is also the wrong bar. The Qwen3 patch accepts an extra
    keyword-only dense_weights that stock does not have, which is harmless:
    the block calls self.experts(hidden, selected_experts, routing_weights) and
    never passes it. What has to hold is that the parameters the caller
    actually uses are all accepted, in the same positions.
    """
    import importlib
    import inspect

    patch = getattr(importlib.import_module("usaf." + module), fn)
    real = getattr(importlib.import_module(realmod), cls)
    stock = getattr(real, "_usaf_original", None) or real.forward
    assert stock is not patch, (
        "the comparison is against the patch itself - the test is vacuous")
    want = list(inspect.signature(stock).parameters)
    got = list(inspect.signature(patch).parameters)
    assert got[:len(want)] == want, (
        f"stock is called with {want}, the patch offers {got}")


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
