import torch
import torch.nn as nn

from usaf.model_factory import MoEConfig, resolve_expert_layout
from usaf.train import _weights_to_load

N_LAYERS, N_EXP, INTER, HID = 3, 4, 8, 6
BLOCK = """model.layers.{i}.block_sparse_moe"""
CONTAINER = BLOCK + """.experts"""


def _cfg():
    return MoEConfig(
        model_path="""x""",
        num_layers=N_LAYERS,
        num_experts=N_EXP,
        expert_prefix=CONTAINER,
        expert_param_names=["""gate_up_proj""", """down_proj"""],
        router_path=""".block_sparse_moe.router.weight""",
    )


def _model(inlined):
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList([nn.Module() for _ in range(N_LAYERS)])
    for layer in m.model.layers:
        b = nn.Module()
        layer.block_sparse_moe = b
        if inlined:
            b.input_linear = nn.Module()
            b.input_linear.weight = nn.Parameter(
                torch.zeros(N_EXP, 2 * INTER, HID))
            b.output_linear = nn.Module()
            b.output_linear.weight = nn.Parameter(
                torch.zeros(N_EXP, HID, INTER))
        else:
            b.experts = nn.Module()
            b.experts.gate_up_proj = nn.Parameter(
                torch.zeros(N_EXP, 2 * INTER, HID))
            b.experts.down_proj = nn.Parameter(
                torch.zeros(N_EXP, HID, INTER))
        b.router = nn.Module()
        b.router.weight = nn.Parameter(torch.zeros(N_EXP, HID))
    return m


def _expert_tensors(cfg):
    return {
        f"""{cfg.expert_prefix.format(i=li)}.{pn}"""
        for li in range(cfg.num_layers)
        for pn in cfg.expert_param_names
    }


def test_the_routers_are_not_mistaken_for_expert_weights_when_inlined():
    # A release that inlines the experts puts the router in the same module as
    # the expert weights. The skip used to be "under one of the expert
    # modules", which is right when the experts are a module of their own and
    # wrong here: the router was skipped along with the experts, never loaded,
    # and the first backward died on the meta tensor it was left on. The count
    # said 146 of 170, which is 24 - one router per layer - and nothing said
    # what the 24 were.
    cfg = resolve_expert_layout(_model(True), _cfg())
    tensors = _expert_tensors(cfg)
    routers = {f"""{BLOCK.format(i=li)}.router.weight"""
              for li in range(N_LAYERS)}

    assert not (tensors & routers)
    assert len(tensors) == N_LAYERS * 2
    assert len(routers) == N_LAYERS


def test_the_routers_are_kept_when_the_experts_are_a_separate_module():
    cfg = resolve_expert_layout(_model(False), _cfg())
    routers = {f"""{BLOCK.format(i=li)}.router.weight"""
              for li in range(N_LAYERS)}
    assert not (_expert_tensors(cfg) & routers)


def _weight_map(inlined):
    """What the checkpoint holds, in both layouts.

    Same 218 keys either way. The experts are stored as input_linear.weight and
    output_linear.weight and the router as router.layer.weight, and a release
    that renames them on the way in has not renamed them in the file.
    """
    out = {}
    for li in range(N_LAYERS):
        stem = f"model.layers.{li}.block_sparse_moe"
        if inlined:
            out[f"{stem}.input_linear.weight"] = f"{stem}.input_linear.weight"
            out[f"{stem}.output_linear.weight"] = f"{stem}.output_linear.weight"
        else:
            out[f"{stem}.input_linear.weight"] = f"{stem}.experts.gate_up_proj"
            out[f"{stem}.output_linear.weight"] = f"{stem}.experts.down_proj"
        out[f"{stem}.router.layer.weight"] = f"{stem}.router.weight"
    return out


def test_the_routers_are_loaded_when_the_experts_are_inlined():
    # The real question, asked of the real function. The exclusion used to be
    # "is this name under one of the expert modules", which is wrong whenever the
    # experts are not a module of their own: the router is in the same place, and
    # it was skipped along with the experts, never loaded, left on the meta device
    # it had been built with. The count said 146 of 170 - 24 short, one router per
    # layer - and nothing said which 24.
    model = _model(True)
    cfg = resolve_expert_layout(model, _cfg())
    mp = dict(model.named_parameters())
    wf = _weight_map(True)

    plan = _weights_to_load(wf, mp, _expert_tensors(cfg))
    routers = {f"{BLOCK.format(i=li)}.router.weight" for li in range(N_LAYERS)}
    assert {n for n in plan.values() if n in routers} == routers

    # And the experts themselves are still skipped, or nothing would be sparse.
    assert not ({n for n in plan.values() if n in _expert_tensors(cfg)})


def test_the_routers_are_loaded_when_the_experts_are_a_separate_module():
    model = _model(False)
    cfg = resolve_expert_layout(model, _cfg())
    mp = dict(model.named_parameters())

    plan = _weights_to_load(_weight_map(False), mp, _expert_tensors(cfg))
    routers = {f"{BLOCK.format(i=li)}.router.weight" for li in range(N_LAYERS)}
    assert {n for n in plan.values() if n in routers} == routers


def test_the_same_number_of_weights_is_loaded_in_both_layouts():
    # Three keys a layer in this fixture: two experts and one router. A layout that
    # quietly drops the routers still produces a number, and on the real model that
    # number is the only place it would show up - 146 instead of 170, twenty-four
    # short, one router per layer, and nothing in the log saying which.
    for inlined in (True, False):
        model = _model(inlined)
        cfg = resolve_expert_layout(model, _cfg())
        wf = _weight_map(inlined)
        plan = _weights_to_load(wf, dict(model.named_parameters()),
                                _expert_tensors(cfg))

        assert len(wf) == 3 * N_LAYERS
        assert len(plan) == N_LAYERS, (inlined, len(plan))
