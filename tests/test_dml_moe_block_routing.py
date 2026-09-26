"""The block is the router, and the block was never tested.

the existing checks in this suite call the expert container directly:
experts(hidden, idx, weights). That skips the layer that decides which experts
fire and with what weight, which is where a patched forward goes wrong in the
way that matters - it accepts the arguments and then mixes the experts up. The
file holding these replacements says exactly that, and the test called the
container.

One thing this had to get right. The stock container forward is not the
function its signature suggests: transformers decorates it into a
(*args, **kwargs) wrapper that dispatches on config._experts_implementation,
with @wraps on the original. inspect.signature follows __wrapped__, so it
reports a three-argument signature that the real call path never uses.

And the DML block calls the container with dense_weights=, which only the DML
container accepts. So the two have to be installed together - calling the DML
alone raises "got an unexpected keyword argument dense_weights". Both halves are
patched here, the way patch_*_for_dml does it, and the stock run is compared
against that.
"""
import importlib
import types

import pytest
import torch

FAMILIES = [
    ("qwen3moe_dml", "dml_qwen3_moe_block_forward",
     "transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeSparseMoeBlock",
     "dml_qwen3_experts_forward", "Qwen3MoeExperts",
     "patch_qwen3moe_for_dml", "unpatch_qwen3moe_for_dml"),
    ("mixtral_dml", "dml_mixtral_moe_block_forward",
     "transformers.models.mixtral.modeling_mixtral", "MixtralSparseMoeBlock",
     "dml_mixtral_experts_forward", "MixtralExperts",
     "patch_mixtral_for_dml", "unpatch_mixtral_for_dml"),
    ("olmoe_dml", "dml_moe_block_forward",
     "transformers.models.olmoe.modeling_olmoe", "OlmoeSparseMoeBlock",
     "dml_experts_forward", "OlmoeExperts",
     "patch_olmoe_for_dml", "unpatch_olmoe_for_dml"),
]


def _cfg():
    return types.SimpleNamespace(
        num_experts=4,
        num_experts_per_tok=2,
        num_local_experts=4,
        num_local_experts_per_tok=2,
        hidden_size=32,
        moe_intermediate_size=24,
        intermediate_size=24,
        hidden_act="silu",
        _experts_implementation="eager",
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
        router_jitter_noise=0.0,
        rms_norm_eps=1e-6,
    )


@pytest.mark.parametrize("module,fn,realmod,cls,efn,ecls,patch,unpatch", FAMILIES)
def test_the_patched_block_routes_the_way_the_stock_one_does(
        module, fn, realmod, cls, efn, ecls, patch, unpatch):
    real = importlib.import_module(realmod)
    usaf = importlib.import_module("usaf." + module)
    block_cls = getattr(real, cls)

    torch.manual_seed(0)
    block = block_cls(_cfg()).to(torch.float32).eval()
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.05)
    hs = torch.randn(5, 1, 32)

    with torch.no_grad():
        want = block(hs)
        getattr(usaf, patch)()
        try:
            got = block(hs)
        finally:
            getattr(usaf, unpatch)()

    err = (got - want.float()).abs().max().item()
    scale = max(want.abs().max().item(), 1e-6)
    assert err / scale < 0.05, (f"{module}: max abs error {err:.3e} on output "
                                  f"of magnitude {scale:.3f}")


@pytest.mark.parametrize("module,fn,realmod,cls,efn,ecls,patch,unpatch", FAMILIES)
def test_the_patched_block_is_not_a_pass_through(
        module, fn, realmod, cls, efn, ecls, patch, unpatch):
    """Otherwise the comparison above is satisfied by returning the input."""
    real = importlib.import_module(realmod)
    usaf = importlib.import_module("usaf." + module)
    block_cls = getattr(real, cls)

    torch.manual_seed(1)
    block = block_cls(_cfg()).to(torch.float32).eval()
    hs = torch.randn(5, 1, 32)
    getattr(usaf, patch)()
    try:
        with torch.no_grad():
            out = block(hs)
    finally:
        getattr(usaf, unpatch)()
    assert out.shape == hs.shape, (out.shape, hs.shape)
    assert not torch.allclose(out.float(), hs), "the block returned its input"


@pytest.mark.parametrize("module,fn,realmod,cls,efn,ecls,patch,unpatch", FAMILIES)
def test_the_dml_block_needs_the_dml_container_and_not_only_the_stock_one(
        module, fn, realmod, cls, efn, ecls, patch, unpatch):
    """The two halves are a matched pair, and this is why.

    The DML block hands the container a dense_weights= argument. The stock
    container, reached through the transformers dispatch wrapper, has no such
    parameter, so the block on its own does not run. Nothing in the training
    path can get this wrong today because setup_device patches both, but the
    pairing is invisible from either function alone.
    """
    real = importlib.import_module(realmod)
    usaf = importlib.import_module("usaf." + module)
    block_cls = getattr(real, cls)

    torch.manual_seed(2)
    block = block_cls(_cfg()).to(torch.float32).eval()
    hs = torch.randn(3, 1, 32)

    original_block = block_cls.forward
    block_cls.forward = getattr(usaf, fn)
    try:
        with torch.no_grad():
            with pytest.raises(TypeError, match="dense_weights"):
                block(hs)
    finally:
        block_cls.forward = original_block
