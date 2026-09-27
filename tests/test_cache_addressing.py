import torch

from usaf.moe_loader import QuantizedExpertCache

N_LAYERS, N_EXP, INTER, HID = 2, 4, 8, 6


def _entry(shape):
    """A plain tensor entry, which the cache passes through as it is.

    These tests are about which address a tensor answers to, not about how a
    packed int4 payload is unpacked, and a real one would put a dequantiser bug
    in the middle of a lookup test.
    """
    return torch.zeros(*shape, dtype=torch.float16)


def _submodule_payload():
    """The spelling the training machine has."""
    out = {}
    for li in range(N_LAYERS):
        stem = f"model.layers.{li}.block_sparse_moe"
        out[f"{stem}.input_linear.weight"] = _entry((N_EXP, 2 * INTER, HID))
        out[f"{stem}.output_linear.weight"] = _entry((N_EXP, HID, INTER))
    return out


def _fused_payload():
    """The other release: the file says experts.gate_up_proj."""
    out = {}
    for li in range(N_LAYERS):
        stem = f"model.layers.{li}.block_sparse_moe.experts"
        out[f"{stem}.gate_up_proj"] = _entry((N_EXP, 2 * INTER, HID))
        out[f"{stem}.down_proj"] = _entry((N_EXP, HID, INTER))
    return out

def _cache(payload, prefix):
    return QuantizedExpertCache(payload, torch.device("cpu"), max_cached=4,
                                group_size=4, expert_prefix=prefix)


def test_the_container_finds_the_weights_held_in_submodules():
    # The container the trainer resolves to is block_sparse_moe, and the
    # parameters it asks for are input_linear.weight and output_linear.weight.
    # Indexing the file on its last component alone files them under
    # block_sparse_moe.input_linear, the lookup misses, the pre-hook places
    # nothing, and the backward fails on the meta tensor the weights never left.
    cache = _cache(_submodule_payload(), "model.layers.{i}.block_sparse_moe")
    got = cache.get_expert_weights("model.layers.0.block_sparse_moe")
    assert set(got) == {"input_linear.weight", "output_linear.weight"}, got
    assert tuple(got["input_linear.weight"].shape) == (N_EXP, 2 * INTER, HID)


def test_the_submodule_address_still_resolves():
    # The same tensor, addressed the way a layout without submodules would.
    cache = _cache(_submodule_payload(), "model.layers.{i}.block_sparse_moe")
    got = cache.get_expert_weights(
        "model.layers.0.block_sparse_moe.input_linear")
    assert set(got) == {"weight"}


def test_a_fused_layout_resolves_through_the_same_index():
    # Nothing about the index changed between the two releases, because the file
    # is the vocabulary both sides already share.
    cache = _cache(_fused_payload(),
                    "model.layers.{i}.block_sparse_moe.experts")
    got = cache.get_expert_weights(
        "model.layers.0.block_sparse_moe.experts")
    assert set(got) == {"gate_up_proj", "down_proj"}


def test_the_router_and_the_ordinary_weights_are_not_mistaken_for_experts():
    # The index now holds each tensor at several addresses. It must not start
    # answering for names it does not hold.
    cache = _cache(_submodule_payload(), "model.layers.{i}.block_sparse_moe")
    assert cache.get_expert_weights("model.layers.0.block_sparse_moe.router") == {}
