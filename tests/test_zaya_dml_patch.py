"""ZAYA was trained by a run that had no ZAYA patch.

The failure had every symptom of working. detect_model printed the right
architecture, the quantiser wrote the right tensors, the expert modules were
found and the capture installed on them, the loss fell step after step, and
the kernel reported Complete. What it never printed was one active weight:

    Active: 0/0 (0.0000%)
    Optimizer: 0.0MB

The sparse-gradient capture lives inside the patched expert forward, and the
patch only replaced Qwen3MoeExperts and Qwen3MoeSparseMoeBlock. ZAYA has its
own ZayaExperts, a plain nn.Module with no inheritance from either, so no
hook ever fired and TopKImportanceStore.select returned an empty dict. The
1163 non-expert parameters kept training, which is what kept the loss moving.

These tests are the ones that would have caught it, and they are built around
a ZayaExperts stand-in with the attributes and the call convention of the
real one, because a test of a patch is worthless if the container it
exercises is not the container the model has.
"""
import contextlib
import sys
import types

import pytest
import torch
import torch.nn as nn

from usaf.moe_loader import TopKImportanceStore
from usaf.qwen3moe_dml import dml_qwen3_experts_forward
from usaf.zaya_dml import patch_zaya_for_dml, unpatch_zaya_for_dml

HIDDEN, INTER, NEXP = 8, 4, 3
PREFIX = "model.layers.3.mlp.experts"


class FakeZayaExperts(nn.Module):
    """The real ZayaExperts layout, copied field for field.

    gate_up_proj is [E, 2*inter, hidden] and down_proj is [E, hidden, inter],
    which is the Qwen3 fused layout. The stock forward is the one from the fork:
    it routes with one_hot and index_add_, and carries no capture hook.
    """

    def __init__(self):
        super().__init__()
        self.num_experts = NEXP
        self.hidden_dim = HIDDEN
        self.intermediate_dim = INTER
        torch.manual_seed(0)
        self.gate_up_proj = nn.Parameter(torch.randn(NEXP, 2 * INTER, HIDDEN) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(NEXP, HIDDEN, INTER) * 0.1)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """The fork forward verbatim in shape: no capture, dense index_add_."""
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            mask = mask.permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            pos, tok = torch.where(mask[expert_idx])
            cur = hidden_states[tok]
            gate, up = nn.functional.linear(cur, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            cur = self.act_fn(gate) * up
            cur = nn.functional.linear(cur, self.down_proj[expert_idx])
            cur = cur * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final


@contextlib.contextmanager
def patched_forward(cls=FakeZayaExperts):
    """Install the replacement, then put the original *value* back.

    del would be wrong: forward lives in the class dict, so assigning over it
    and then deleting it leaves the class with no forward at all.
    """
    stock = cls.forward
    cls.forward = dml_qwen3_experts_forward
    try:
        yield
    finally:
        cls.forward = stock


@pytest.fixture
def experts_with_capture():
    mod = FakeZayaExperts()
    shapes = {
        f"{PREFIX}.gate_up_proj": (NEXP, 2 * INTER, HIDDEN),
        f"{PREFIX}.down_proj": (NEXP, HIDDEN, INTER),
    }
    store = TopKImportanceStore(shapes, frac=0.05)
    mod._grad_capture = (store, PREFIX)
    yield mod, store
    del mod._grad_capture


def _call(mod, n=12):
    torch.manual_seed(1)
    hs = torch.randn(n, HIDDEN, requires_grad=True)
    idx = torch.randint(0, NEXP, (n, 1))
    w = torch.rand(n, 1) + 0.1
    out = mod(hs, idx, w)
    out.sum().backward()
    return out.detach()


def test_the_stock_zaya_forward_captures_nothing(experts_with_capture):
    """The bug, reproduced: the forward that was really running stores nothing."""
    mod, store = experts_with_capture
    _call(mod)
    assert store.n_captured() == 0, (
        "the stock forward started capturing - the premise of this bug changed"
    )
    assert store.select(0.05) == {}, "select should be empty"


def test_the_patched_zaya_forward_captures_every_expert(experts_with_capture):
    mod, store = experts_with_capture
    with patched_forward():
        _call(mod)
    assert store.n_captured() > 0, "the patched forward captured nothing either"
    sel = store.select(0.05)
    want = {f"{PREFIX}.gate_up_proj", f"{PREFIX}.down_proj"}
    assert set(sel) == want, list(sel)
    for name, idx in sel.items():
        assert idx.numel() > 0, name


def test_patching_agrees_with_the_stock_forward(experts_with_capture):
    """The replacement has to be the same function of its inputs.

    Both loop over the experts, but the stock one gathers only the tokens that
    hit each expert while the replacement runs every expert over every token
    with a zero mask. If they disagree, USAF is training a model the user did
    not choose, and every other test still passes.
    """
    mod, _ = experts_with_capture
    stock = _call(mod)
    with patched_forward():
        patched = _call(mod)
    denom = stock.abs().max().clamp_min(1e-6)
    assert (stock - patched).abs().max() / denom < 0.05, (
        "patched and stock disagree beyond fp16 noise"
    )


def test_the_patch_is_reversible_and_absent_without_the_fork():
    """setup_device must not require a fork to import usaf."""
    try:
        from transformers.models.zaya import modeling_zaya as mz
    except ImportError:
        assert patch_zaya_for_dml() is None
        unpatch_zaya_for_dml()
        return
    stock = mz.ZayaExperts.forward
    assert patch_zaya_for_dml() is not None
    assert mz.ZayaExperts.forward is not stock, "the fork is installed, no patch"
    unpatch_zaya_for_dml()
    assert mz.ZayaExperts.forward is stock, "unpatch did not restore the original"
    assert not hasattr(mz.ZayaExperts, "_usaf_original")


def test_setup_device_actually_installs_the_zaya_patch(monkeypatch):
    """The wiring, not the patch function.

    Every other test here drives dml_qwen3_experts_forward directly, so all of
    them pass whether or not setup_device calls the Zaya patch. That is the
    trap this one closes: the patch was written, unit-tested, and still never
    reached the model, because nothing checked the one line joining them.
    """

    class FakeExperts(torch.nn.Module):
        def forward(self, hidden_states, top_k_index, top_k_weights):
            return hidden_states

    fake = types.ModuleType("transformers.models.zaya.modeling_zaya")
    fake.ZayaExperts = FakeExperts
    pkg = types.ModuleType("transformers.models.zaya")
    pkg.modeling_zaya = fake
    monkeypatch.setitem(sys.modules, "transformers.models.zaya", pkg)
    monkeypatch.setitem(
        sys.modules, "transformers.models.zaya.modeling_zaya", fake
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    from usaf.train import TrainConfig, setup_device

    stock = FakeExperts.forward
    try:
        setup_device(TrainConfig(model_path="m", dataset_path="d", use_cuda=False))
        assert FakeExperts.forward is not stock, (
            "setup_device returned without patching ZayaExperts: a ZAYA run"
            " would train zero experts and still show a falling loss"
        )
    finally:
        FakeExperts.forward = stock
