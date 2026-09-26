"""A process-wide patch that cannot be taken back is a patch that cannot be
scoped.

patch_qwen3moe_for_dml, patch_olmoe_for_dml and patch_mixtral_for_dml each
replace a forward on a transformers class, and the docstring says so: "affects
all model instances". setup_device calls all three, and there was no way to
undo it.

That is not only untidy. A test suite that runs the trainer in-process leaves
every later test running the dense-masked forward instead of the stock one, so
the tests after it are quietly testing something else. It showed up as a
dtype error in a streaming test that passes on its own - a failure that looks
like a bug in the streaming path and is actually a bug in the test suite.
"""
import pytest


@pytest.fixture()
def qwen3():
    # Start from a clean slate. patch_*_for_dml is a process-wide assignment
    # and the other test file that touches these classes installs it too, so
    # whichever order pytest collects in, the starting point has to be
    # established here rather than assumed.
    from usaf.mixtral_dml import unpatch_mixtral_for_dml
    from usaf.olmoe_dml import unpatch_olmoe_for_dml
    from usaf.qwen3moe_dml import unpatch_qwen3moe_for_dml
    unpatch_qwen3moe_for_dml()
    unpatch_olmoe_for_dml()
    unpatch_mixtral_for_dml()
    from transformers.models.qwen3_moe import modeling_qwen3_moe
    cls = modeling_qwen3_moe.Qwen3MoeExperts
    block = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
    before = (cls.forward, block.forward)
    yield before
    unpatch_qwen3moe_for_dml()
    unpatch_olmoe_for_dml()
    unpatch_mixtral_for_dml()
    cls.forward, block.forward = before


def test_patching_changes_the_forward_and_unpatching_puts_it_back(qwen3):
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    from usaf.qwen3moe_dml import dml_qwen3_experts_forward, patch_qwen3moe_for_dml
    cls = modeling_qwen3_moe.Qwen3MoeExperts
    original = cls.forward
    patch_qwen3moe_for_dml()
    assert cls.forward is dml_qwen3_experts_forward
    from usaf.qwen3moe_dml import unpatch_qwen3moe_for_dml
    unpatch_qwen3moe_for_dml()
    assert cls.forward is original


def test_the_sibling_module_is_restored_too(qwen3):
    from transformers.models.qwen3_moe import modeling_qwen3_moe
    block = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
    original = block.forward
    from usaf.qwen3moe_dml import patch_qwen3moe_for_dml, unpatch_qwen3moe_for_dml
    patch_qwen3moe_for_dml()
    unpatch_qwen3moe_for_dml()
    assert block.forward is original


def test_patching_twice_does_not_lose_the_original(qwen3):
    """Otherwise the second patch records the first patch as the original, and
    unpatching hands back the DML forward instead of the stock one."""
    from transformers.models.qwen3_moe import modeling_qwen3_moe
    cls = modeling_qwen3_moe.Qwen3MoeExperts
    original = cls.forward
    from usaf.qwen3moe_dml import (
        dml_qwen3_experts_forward,
        patch_qwen3moe_for_dml,
        unpatch_qwen3moe_for_dml,
    )
    patch_qwen3moe_for_dml()
    patch_qwen3moe_for_dml()
    assert cls.forward is dml_qwen3_experts_forward
    unpatch_qwen3moe_for_dml()
    assert cls.forward is original, "unpatch handed back the patch, not the code"


def test_unpatching_twice_is_harmless(qwen3):
    from usaf.qwen3moe_dml import unpatch_qwen3moe_for_dml
    unpatch_qwen3moe_for_dml()
    unpatch_qwen3moe_for_dml()


def test_the_other_two_families_can_be_reverted_too(qwen3):
    from transformers.models.mixtral import modeling_mixtral
    from transformers.models.olmoe import modeling_olmoe

    from usaf.mixtral_dml import patch_mixtral_for_dml, unpatch_mixtral_for_dml
    from usaf.olmoe_dml import patch_olmoe_for_dml, unpatch_olmoe_for_dml
    m_before = modeling_mixtral.MixtralExperts.forward
    o_before = modeling_olmoe.OlmoeExperts.forward
    try:
        patch_mixtral_for_dml()
        patch_olmoe_for_dml()
        assert modeling_mixtral.MixtralExperts.forward is not m_before
        assert modeling_olmoe.OlmoeExperts.forward is not o_before
        unpatch_mixtral_for_dml()
        unpatch_olmoe_for_dml()
        assert modeling_mixtral.MixtralExperts.forward is m_before
        assert modeling_olmoe.OlmoeExperts.forward is o_before
    finally:
        unpatch_mixtral_for_dml()
        unpatch_olmoe_for_dml()
