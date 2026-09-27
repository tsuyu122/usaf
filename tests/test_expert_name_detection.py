"""
The name detector was a table, and the table was wrong for every model
that was not the one it was written for.

_detect_param_names returned a fixed triple - mlp.experts, gate_up_proj and
down_proj, router at .mlp.gate.weight - and ignored the config it was handed.
ZAYA1-8B worked anyway, and it worked for the wrong reason: its names happen
to match. GraniteMoe does not, and it is a plain MoE with the fused expert
container every other supported family has:

    prefix  model.layers.{i}.block_sparse_moe.experts   (not mlp.experts)
    router  .block_sparse_moe.router.weight             (not .mlp.gate.weight)

Two things had to be discovered rather than assumed, and both are quiet when
wrong. The expert tensors are stored on disk as input_linear / output_linear
and renamed on load, so a detector reading the file for the loaded names
matches neither. And the router is router.layer.weight on disk but
router.weight on the module, so a detector reading the file for a path that
is looked up in the module resolves to nothing at runtime, with no error -
the ZAYA failure in a new place, where the run starts, prints its
architecture and trains nothing.

These build fixtures with the real checkpoint layouts rather than a single
one that agrees, because a table of one shape cannot detect a model of
another.
"""

import json

from usaf.model_factory import (
    _expert_names_from_index,
    _keys_from_single_safetensors,
    _loaded_router_name,
)

LAYER0 = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.mlp.experts.gate_up_proj",
    "model.layers.0.mlp.experts.down_proj",
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.post_attention_layernorm.weight",
]


def _sharded(tmp_path, names, shard=2):
    """A checkpoint split into shards with an index."""
    weight_map = {n: f"model-00001-of-{shard:05d}.safetensors"
               for n in names}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return str(tmp_path)


def _single(tmp_path, names):
    """A one-file checkpoint, written as a real safetensors header.

    data_offsets are not decoration: the safetensors spec requires them, and a
    header without them is a file the real library will not open. A fixture that
    is more permissive than the format it stands for tests nothing.
    """
    import struct

    off = 0
    table = {}
    for n in names:
        size = 4
        table[n] = {"dtype": "F32", "shape": [1], "data_offsets": [off, off + size]}
        off += size
    header = json.dumps({**table, "__metadata__": {}}).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * off)
    return str(tmp_path)


def test_qwen_layout_is_read_from_the_index(tmp_path):
    got = _expert_names_from_index(_sharded(tmp_path, LAYER0))
    assert got == ("model.layers.{i}.mlp.experts",
                   ["gate_up_proj", "down_proj"],
                   ".mlp.gate.weight"), got


def test_granite_layout_is_read_and_not_assumed(tmp_path):
    # block_sparse_moe, not mlp; router.weight, not gate.weight. A fixed
    # table returns the Qwen answer here and every name in it is wrong.
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.block_sparse_moe.input_linear.weight",
        "model.layers.0.block_sparse_moe.output_linear.weight",
        "model.layers.0.block_sparse_moe.router.layer.weight",
    ]
    got = _expert_names_from_index(_single(tmp_path, names))
    assert got is not None, "Granite has no index; the single-file path "\
        "was never taken and the model is undetectable"
    prefix, names_out, router = got
    assert prefix == "model.layers.{i}.block_sparse_moe.experts", prefix
    assert names_out == ["gate_up_proj", "down_proj"], names_out
    assert router == ".block_sparse_moe.router.weight", router


def test_the_router_on_disk_is_not_the_router_on_the_module():
    # This is the quiet one. The file says router.layer.weight, the module
    # says router.weight, and the trainer looks names up in the module.
    assert _loaded_router_name("router.layer.weight") == \
        "router.weight"
    assert _loaded_router_name("router.weight") == "router.weight"
    assert _loaded_router_name("gate.weight") == "gate.weight"


def test_an_expert_tensor_is_never_mistaken_for_the_router(tmp_path):
    # input_linear has no "expert" in its name, so a name-only filter
    # returns it and the router path points at a weight matrix.
    names = [
        "model.layers.0.block_sparse_moe.input_linear.weight",
        "model.layers.0.block_sparse_moe.output_linear.weight",
        "model.layers.0.block_sparse_moe.router.weight",
    ]
    _prefix, _n, router = _expert_names_from_index(
        _single(tmp_path, names))
    assert router == ".block_sparse_moe.router.weight", router


def test_layers_that_disagree_are_not_a_prefix(tmp_path):
    # One layer under mlp and another under block_sparse_moe is not a
    # naming convention, it is a model this does not understand, and
    # picking the majority silently trains half of it.
    names = LAYER0 + [
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.block_sparse_moe.experts.down_proj",
    ]
    assert _expert_names_from_index(_sharded(tmp_path, names)) is None


def test_a_model_with_no_experts_is_not_moe(tmp_path):
    names = ["model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.mlp.up_proj.weight"]
    assert _expert_names_from_index(_sharded(tmp_path, names)) is None
    assert _expert_names_from_index(_single(tmp_path, names)) is None


def test_no_checkpoint_is_not_a_crash(tmp_path):
    assert _expert_names_from_index(str(tmp_path)) is None
    assert _keys_from_single_safetensors(str(tmp_path)) == []


def test_several_shards_without_an_index_is_refused(tmp_path):
    for i in range(2):
        (tmp_path / f"model-0000{i + 1}.safetensors").write_bytes(b"")
    assert _keys_from_single_safetensors(str(tmp_path)) == []


def test_a_truncated_header_is_refused(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\xff\xff\xff\xff")
    assert _keys_from_single_safetensors(str(tmp_path)) == []


def test_the_detector_is_actually_reached(tmp_path):
    # Every other test here calls the helper directly, so all of them pass with
    # the detector disabled at the call site - the one place where a regression
    # would hide. This goes through _detect_param_names, which is what the
    # trainer calls.
    from usaf.model_factory import _detect_param_names

    got = _detect_param_names(None, _sharded(tmp_path, LAYER0))
    assert got == ("model.layers.{i}.mlp.experts",
                   ["gate_up_proj", "down_proj"],
                   ".mlp.gate.weight"), got


def test_a_granite_checkpoint_reaches_the_same_call_site(tmp_path):
    # The same question, asked of the other layout, through the real entry point.
    from usaf.model_factory import _detect_param_names

    names = [
        "model.layers.0.block_sparse_moe.input_linear.weight",
        "model.layers.0.block_sparse_moe.output_linear.weight",
        "model.layers.0.block_sparse_moe.router.layer.weight",
    ]
    got = _detect_param_names(None, _single(tmp_path, names))
    assert got == ("model.layers.{i}.block_sparse_moe.experts",
                   ["gate_up_proj", "down_proj"],
                   ".block_sparse_moe.router.weight"), got


def test_a_path_with_no_checkpoint_falls_back_and_says_so(tmp_path):
    # With nothing to read there is no evidence, and the Qwen names are the
    # only ones available. The point is that this is a fallback and not a
    # second, equally-weighted answer: Granite through this path would train
    # nothing, and the caller has to be able to tell that from a real read.
    from usaf.model_factory import _detect_param_names

    assert _detect_param_names(None, str(tmp_path)) == (
        "model.layers.{i}.mlp.experts",
        ["gate_up_proj", "down_proj"],
        ".mlp.gate.weight",
    )
