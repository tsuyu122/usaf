"""setup_quantized_streaming, which the trainer does not use and nothing called.

The trainer builds its own QuantizedExpertCache and its own pre-hooks, so this
function - exported from the package, 133 lines of hook installation, gradient
checkpointing and cache wiring - had never run anywhere. A second
implementation of the same thing, kept beside the one that is actually on the
path, is exactly where a fixed bug goes to be reintroduced.

So: build a model, quantize its experts, stream it, and check the answer is the
answer the dense model gives. A hook that does not fire, or fires with the
wrong name, produces zeros and no error.
"""
import os

import pytest
import torch

from usaf.moe_loader import get_quantized_cache, setup_quantized_streaming

E2E = r"C:\Users\hm\Projects\e2e"
MODEL = os.path.join(E2E, "tiny-moe")
Q4 = os.path.join(E2E, "tiny-moe-q4", "experts_q4.pt")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(MODEL) and os.path.exists(Q4)),
    reason="e2e fixture absent")


def _load(dtype=torch.float32):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    m.eval()
    return m


@pytest.fixture(scope="module")
def q_dict():
    return torch.load(Q4, map_location="cpu", weights_only=False)


def test_the_hooks_are_installed_on_the_expert_modules(q_dict):
    m = _load()
    n_experts = sum(1 for name, _ in m.named_modules()
                    if name.endswith(".mlp.experts"))
    assert n_experts, "the fixture model has no expert modules"
    setup_quantized_streaming(m, q_dict, torch.device("cpu"),
                              max_cached_experts=2, verbose=False)
    hooked = [mod for _, mod in m.named_modules() if mod._forward_pre_hooks]
    assert len(hooked) == n_experts, (len(hooked), n_experts)


def test_the_cache_it_installed_is_reachable_through_the_model(q_dict):
    m = _load()
    setup_quantized_streaming(m, q_dict, torch.device("cpu"),
                              max_cached_experts=2, verbose=False)
    cache = get_quantized_cache(m)
    assert cache is not None, "get_quantized_cache found nothing"
    assert cache.cached_experts == [], "nothing was dequantized before a forward"


def test_a_forward_produces_the_answers_of_the_dense_model(q_dict):
    """The point of the whole thing. A hook that does not fire gives zeros."""
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (1, 12))
    dense = _load()
    with torch.no_grad():
        want = dense(input_ids=ids).logits.float()

    streamed = _load()
    setup_quantized_streaming(streamed, q_dict, torch.device("cpu"),
                              max_cached_experts=2, verbose=False)
    with torch.no_grad():
        got = streamed(input_ids=ids).logits.float()

    assert not torch.allclose(got, want), (
        "the streamed model produced exactly the dense answer - the experts "
        "were never dequantized, or the hooks never fired")
    err = (got - want).abs().max().item()
    scale = want.abs().max().item()
    assert err / max(scale, 1e-6) < 0.25, (
        f"max abs error {err:.4f} on logits of magnitude {scale:.3f}")


def test_the_expert_parameters_are_released_after_the_forward(q_dict):
    """Cleared in the post-hook, so a module does not hold its experts forever."""
    m = _load()
    setup_quantized_streaming(m, q_dict, torch.device("cpu"),
                              max_cached_experts=2, verbose=False)
    experts = [mod for name, mod in m.named_modules()
               if name.endswith(".mlp.experts")]
    held = sum(len(mod._parameters) for mod in experts)
    with torch.no_grad():
        m(input_ids=torch.randint(0, 100, (1, 8)))
    after = sum(len(mod._parameters) for mod in experts)
    assert held == 0, f"the expert modules already hold {held} parameters"
    assert after == 0, f"{after} parameters left attached after the forward"
