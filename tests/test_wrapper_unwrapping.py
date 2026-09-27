import pytest
import torch
import torch.nn as nn

from usaf.train import _expert_module_names, _expert_modules_by_name, _install_expert_hooks

N_LAYERS, N_EXP, INTER, HID = 2, 4, 8, 6
PREFIX = """model.layers.{i}.block_sparse_moe.experts"""


class _Cfg:
    num_layers = N_LAYERS
    expert_prefix = PREFIX


def _build():
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList([nn.Module() for _ in range(N_LAYERS)])
    for layer in m.model.layers:
        layer.block_sparse_moe = nn.Module()
        layer.block_sparse_moe.experts = _Experts()
    return m




class _Experts(nn.Module):
    """A stand-in with a forward, so calling the module exercises the hooks."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(N_EXP, 2 * INTER, HID))
        self.down_proj = nn.Parameter(torch.zeros(N_EXP, HID, INTER))

    def forward(self, x):
        return x

class _Wrapper(nn.Module):
    """A wrapper that prefixes every name it reports, as DataParallel does.

    It is not a DataParallel on purpose. The class it is here resolves to on
    this machine is the class the check names, so a test built that way passes
    with the isinstance form in place and says nothing about the training
    machine, where the wrapper is a different class of the same name. What has
    to hold is the behaviour: a wrapper whose names do not match still
    resolves.
    """

    def __init__(self, inner):
        super().__init__()
        self.module = inner


class _Cache:
    """Enough of QuantizedExpertCache to fire the hook it feeds."""

    def __init__(self):
        self.asked = []

    def get_expert_weights(self, name):
        self.asked.append(name)
        return {'gate_up_proj': torch.ones(1), 'down_proj': torch.ones(1)}


def _found(model):
    return _expert_modules_by_name(model, _expert_module_names(_Cfg()))


def test_the_experts_are_found_through_a_wrapper_that_prefixes_every_name():
    found = _found(_Wrapper(_build()))
    assert len(found) == N_LAYERS
    assert all(name.startswith('model.layers.') for name in found)


def test_a_model_that_needs_no_unwrapping_is_untouched():
    found = _found(_build())
    assert len(found) == N_LAYERS
    assert all(name.startswith('model.layers.') for name in found)


def test_the_same_modules_come_back_either_way():
    # The importance capture and the hook installation have to agree on which
    # modules they are. Unwrapping changes which names are looked up and
    # nothing else, because the wrapper shares the objects themselves.
    assert set(_found(_build())) == set(_found(_Wrapper(_build())))


def test_the_hooks_land_on_the_experts_through_a_wrapper():
    # The lookup on its own is not what was broken and is not what is worth
    # asserting here: it was already correct, and a test on it passes with the
    # installation reverted to enumerating the wrapper. This calls the code
    # that runs.
    inner = _build()
    n = _install_expert_hooks(
        _Wrapper(inner), set(_expert_module_names(_Cfg())), _Cache())

    assert n == N_LAYERS
    for name in _expert_module_names(_Cfg()):
        mod = dict(inner.named_modules())[name]
        assert mod._forward_pre_hooks, name
        assert mod._forward_hooks, name


def test_the_hooks_fill_the_cleared_parameters_and_empty_them_again():
    # A forward through the module is the only thing that proves the pair
    # works: the pre-hook puts the weights in and the post-hook takes them
    # back out, so the cache is not left holding a live parameter set.
    inner = _build()
    cache = _Cache()
    _install_expert_hooks(inner, set(_expert_module_names(_Cfg())), cache)
    mod = dict(inner.named_modules())[
        'model.layers.0.block_sparse_moe.experts']

    assert len(mod._parameters) == 0
    # Installing asked the cache for every module; only the forward should be
    # counted from here.
    cache.asked.clear()
    mod(torch.zeros(1))
    assert set(cache.asked) == {'model.layers.0.block_sparse_moe.experts'}
    assert len(mod._parameters) == 0


def test_a_model_with_no_experts_says_what_it_found():
    # A model with no experts is not a silent zero, and the message has to
    # name what it unwrapped, what it wanted and what it actually found - a
    # list of missing names on its own is what sent this debugging the wrong
    # way, and the empty match list was the only thing that gave it away.
    with pytest.raises(SystemExit) as exc:
        _install_expert_hooks(
            _Wrapper(nn.Module()), set(_expert_module_names(_Cfg())), _Cache())
    text = str(exc.value)
    assert 'model.layers.0.block_sparse_moe.experts' in text
    assert '_Wrapper' in text
    assert 'expert-like names actually present: []' in text
