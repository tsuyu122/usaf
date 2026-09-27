import pytest
import torch
import torch.nn as nn

from usaf.train import _expert_module_names, _install_expert_hooks

N_LAYERS, N_EXP, INTER, HID = 2, 4, 8, 6
PREFIX = 'model.layers.{i}.block_sparse_moe.experts'





class _Cache:
    """Enough of QuantizedExpertCache to fire the hook it feeds."""

    def __init__(self):
        self.asked = []

    def get_expert_weights(self, name):
        self.asked.append(name)
        return {
            'input_linear.weight': torch.ones(N_EXP, 2 * INTER, HID),
            'output_linear.weight': torch.ones(N_EXP, HID, INTER),
        }

class _SubmoduleExperts(nn.Module):
    """The layout the training machine has: the experts are modules in their

    own right, each holding a weight, rather than Parameters of the container.
    """

    def __init__(self):
        super().__init__()
        self.input_linear = nn.Module()
        self.input_linear.weight = nn.Parameter(
            torch.zeros(N_EXP, 2 * INTER, HID))
        self.output_linear = nn.Module()
        self.output_linear.weight = nn.Parameter(
            torch.zeros(N_EXP, HID, INTER))

    def forward(self, x):
        # Reads the submodules, which is the whole point: a hook that wrote
        # the same keys onto the container would leave this reading the
        # original tensor, still on meta.
        self.seen = self.input_linear.weight.detach().clone()
        return x


def _submodule_model():
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList([nn.Module() for _ in range(N_LAYERS)])
    for layer in m.model.layers:
        layer.block_sparse_moe = nn.Module()
        layer.block_sparse_moe.experts = _SubmoduleExperts()
    return m


class _SubCfg:
    num_layers = N_LAYERS
    expert_prefix = PREFIX


def test_experts_inside_submodules_get_filled_where_the_forward_reads_them():
    inner = _submodule_model()
    cache = _Cache()
    n = _install_expert_hooks(inner, set(_expert_module_names(_SubCfg())), cache)
    assert n == N_LAYERS

    mod = dict(inner.named_modules())['model.layers.0.block_sparse_moe.experts']
    # Before any forward the submodule holds nothing at all: the weight is not
    # there as an attribute, which is the state the run has to recover from.
    with pytest.raises(AttributeError):
        mod.input_linear.weight

    mod.seen = None
    mod(torch.zeros(1))
    after = mod.seen
    assert mod.seen is not None

    # The forward reads input_linear.weight. If the hook had written onto the
    # container instead, this is unchanged - and the backward would fail on the
    # meta device exactly as it did on the training machine.
    assert after.abs().sum() > 0
    assert after.abs().sum() > 0


def test_submodule_experts_are_empty_again_after_the_forward():
    inner = _submodule_model()
    _install_expert_hooks(inner, set(_expert_module_names(_SubCfg())), _Cache())
    mod = dict(inner.named_modules())['model.layers.0.block_sparse_moe.experts']

    assert len(mod.input_linear._parameters) == 0
    assert len(mod.output_linear._parameters) == 0
    mod(torch.zeros(1))
    assert len(mod.input_linear._parameters) == 0
    assert len(mod.output_linear._parameters) == 0
