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


def test_the_router_name_is_collapsed_in_a_full_path_too():
    # The detector is handed a suffix, but the weight loader walks whole
    # checkpoint keys. A collapse that only works on a bare suffix leaves every
    # router on the meta device, and the run dies naming a weight that was in the
    # file all along.
    full = "model.layers.7.block_sparse_moe.router.layer.weight"
    assert _loaded_router_name(full) == (
        "model.layers.7.block_sparse_moe.router.weight")


def test_a_non_router_name_is_never_collapsed():
    # The layer component is only the routers quirk. Rewriting a name this code
    # does not understand is worse than reporting the name it was given.
    for name in (
        "model.layers.7.block_sparse_moe.router.weight",
        "model.layers.7.mlp.experts.gate_up_proj",
        "model.layers.7.input_layernorm.weight",
        "router.layer_norm.weight",
    ):
        assert _loaded_router_name(name) == name, name


def test_a_granite_router_collapses_where_a_qwen_router_does_not():
    # Both are routers; only one of them is stored under the extra layer.
    assert _loaded_router_name("router.layer.weight") == "router.weight"
    assert _loaded_router_name("mlp.gate.weight") == "mlp.gate.weight"

def test_the_module_name_wins_when_the_model_has_it():
    # The loader asks the model which spelling it has, because the answer
    # differs by release: some transformers flatten router.layer.weight to
    # router.weight on the module, some keep both. Collapsing unconditionally
    # fixes the first and breaks the second, and the second dies with every
    # router still on the meta device, under a name that is in the checkpoint.
    from usaf.train import _resolve_module_name

    disk = 'model.layers.0.block_sparse_moe.router.layer.weight'
    flat = 'model.layers.0.block_sparse_moe.router.weight'

    assert _resolve_module_name(disk, {flat: 0}) == flat
    assert _resolve_module_name(disk, {disk: 0}) == disk


def test_a_name_the_model_has_neither_way_is_not_invented():
    # Resolving must not fabricate a third spelling no loader would find. The
    # membership test downstream is what rejects it.
    from usaf.train import _resolve_module_name

    disk = 'model.layers.0.block_sparse_moe.router.layer.weight'
    assert _resolve_module_name(disk, {}) not in ({}, {disk})


def test_an_untouched_weight_keeps_its_own_name():
    # Everything that is not a router resolves to itself, either way round.
    from usaf.train import _resolve_module_name

    for name in (
        'model.embed_tokens.weight',
        'model.layers.3.mlp.down_proj.weight',
        'model.norm.weight',
    ):
        assert _resolve_module_name(name, {name: 0}) == name
        assert _resolve_module_name(name, {}) == name
import pytest
import torch
import torch.nn as nn

from usaf.model_factory import MoEConfig, resolve_expert_layout

N_LAYERS, N_EXP, INTER, HID = 3, 4, 8, 6
PREFIX = "model.layers.{i}.block_sparse_moe"
WITH_CONTAINER = PREFIX + ".experts"


def _cfg(prefix):
    return MoEConfig(
        model_path="x",
        num_layers=N_LAYERS,
        num_experts=N_EXP,
        expert_prefix=prefix,
        expert_param_names=["gate_up_proj", "down_proj"],
        router_path=".block_sparse_moe.router.weight",
    )


def _block(with_container):
    """A layer block, in the two shapes releases disagree about.

    with_container=True  -> the experts live in a .experts submodule
    with_container=False -> they sit straight in the block
    """
    b = nn.Module()
    b.gate_up_proj = nn.Parameter(torch.zeros(N_EXP, 2 * INTER, HID))
    b.down_proj = nn.Parameter(torch.zeros(N_EXP, HID, INTER))
    b.router = nn.Module()
    b.router.weight = nn.Parameter(torch.zeros(N_EXP, HID))
    if with_container:
        holder = nn.Module()
        holder.gate_up_proj = b.gate_up_proj
        holder.down_proj = b.down_proj
        b.experts = holder
        del b.gate_up_proj
        del b.down_proj
    return b


class _Model(nn.Module):
    def __init__(self, with_container):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Module() for _ in range(N_LAYERS)])
        for layer in self.model.layers:
            layer.block_sparse_moe = _block(with_container)


def test_a_release_that_keeps_the_container_keeps_the_prefix():
    cfg = _cfg(WITH_CONTAINER)
    resolve_expert_layout(_Model(with_container=True), cfg)
    assert cfg.expert_prefix == WITH_CONTAINER


def test_a_release_that_inlines_them_resolves_to_the_block():
    # The checkpoint answers block_sparse_moe for both releases - the key is the
    # same in both - and the loaded module is the only thing that knows. Reading
    # the file picks one of the two shapes and the other finds no expert modules
    # at all, which is a run with no hooks and a loss that falls on the dense
    # weights alone.
    cfg = _cfg(WITH_CONTAINER)
    resolve_expert_layout(_Model(with_container=False), cfg)
    assert cfg.expert_prefix == PREFIX


def test_the_prefix_read_off_the_file_is_corrected_wherever_it_was_wrong():
    # The detector may guess either way depending on the file it read, and the
    # model settles it. Both guesses have to survive being wrong.
    cfg = _cfg(PREFIX)
    resolve_expert_layout(_Model(with_container=True), cfg)
    assert cfg.expert_prefix == WITH_CONTAINER


def test_a_prefix_that_matches_nothing_names_the_tensors_it_found():
    # "no expert modules found" without the list of what is there is a guess,
    # and the guess was wrong four times in a row before this message existed.
    class Empty(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(
                [nn.Module() for _ in range(N_LAYERS)])
            for layer in self.model.layers:
                layer.block_sparse_moe = nn.Module()
                layer.block_sparse_moe.router = nn.Module()
                layer.block_sparse_moe.router.weight = nn.Parameter(
                    torch.zeros(N_EXP, HID))

    with pytest.raises(SystemExit) as exc:
        resolve_expert_layout(Empty(), _cfg(WITH_CONTAINER))
    text = str(exc.value)
    assert "router.weight" in text
    assert WITH_CONTAINER in text
import torch.nn as nn

N_LAYERS, N_EXP, INTER, HID = 3, 4, 8, 6
BLOCK = "model.layers.{i}.block_sparse_moe"
CONTAINER = BLOCK + ".experts"


def _cfg(prefix=CONTAINER, names=("gate_up_proj", "down_proj")):
    return MoEConfig(
        model_path="x",
        num_layers=N_LAYERS,
        num_experts=N_EXP,
        expert_prefix=prefix,
        expert_param_names=list(names),
        router_path=".block_sparse_moe.router.weight",
    )


def _param(parent, name, shape):
    """A tensor registered as name, with or without a submodule of its own."""
    if name.endswith(".weight"):
        sub = nn.Module()
        sub.weight = nn.Parameter(torch.zeros(*shape))
        setattr(parent, name[:-len(".weight")], sub)
    else:
        setattr(parent, name, nn.Parameter(torch.zeros(*shape)))


def _model(container, up, down):
    """The four combinations two releases disagree about, independently."""
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList([nn.Module() for _ in range(N_LAYERS)])
    for layer in m.model.layers:
        b = nn.Module()
        layer.block_sparse_moe = b
        if container:
            b.experts = nn.Module()
        holder = b.experts if container else b
        _param(holder, up, (N_EXP, 2 * INTER, HID))
        _param(holder, down, (N_EXP, HID, INTER))
        b.router = nn.Module()
        b.router.weight = nn.Parameter(torch.zeros(N_EXP, HID))
    return m


CASES = [
    ("container + loaded names", True, "gate_up_proj", "down_proj",
     CONTAINER, ["gate_up_proj", "down_proj"]),
    ("container + stored names", True, "input_linear.weight", "output_linear.weight",
     CONTAINER, ["input_linear.weight", "output_linear.weight"]),
    ("inlined  + loaded names", False, "gate_up_proj", "down_proj",
     BLOCK, ["gate_up_proj", "down_proj"]),
    ("inlined  + stored names", False, "input_linear.weight", "output_linear.weight",
     BLOCK, ["input_linear.weight", "output_linear.weight"]),
]


def test_every_combination_a_release_can_present_resolves():
    for label, container, up, down, want_prefix, want_names in CASES:
        cfg = _cfg()
        resolve_expert_layout(_model(container, up, down), cfg)
        assert cfg.expert_prefix == want_prefix, (label, cfg.expert_prefix)
        assert cfg.expert_param_names == want_names, (label, cfg.expert_param_names)


def test_a_prefix_read_wrong_from_the_file_is_corrected():
    # The detector can hand over either spelling, depending on the file it read.
    # The model settles it, so both guesses have to survive being wrong.
    for guess in (BLOCK, CONTAINER):
        cfg = _cfg(prefix=guess)
        resolve_expert_layout(
            _model(False, "input_linear.weight", "output_linear.weight"), cfg)
        assert cfg.expert_prefix == BLOCK, (guess, cfg.expert_prefix)


def test_a_model_with_no_experts_names_what_it_found():
    import pytest

    class Empty(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(
                [nn.Module() for _ in range(N_LAYERS)])
            for layer in self.model.layers:
                layer.block_sparse_moe = nn.Module()
                layer.block_sparse_moe.router = nn.Module()
                layer.block_sparse_moe.router.weight = nn.Parameter(
                    torch.zeros(N_EXP, HID))

    with pytest.raises(SystemExit) as exc:
        resolve_expert_layout(Empty(), _cfg())
    text = str(exc.value)
    assert "router.weight" in text
    assert CONTAINER in text
