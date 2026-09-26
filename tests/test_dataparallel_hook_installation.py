"""A DataParallel run installed no gradient hooks at all, and said nothing.

nn.DataParallel stores the model as .module and prefixes every name it reports
with "module.". The training loop installs its sparse-gradient hooks by walking
model.named_modules() and matching against the names detect_model reported - and
main wraps the model in DataParallel before handing it over. So the match found
nothing: no expert ever received a capture, the run trained no expert at all,
the loss sat near ln(vocab), and the run printed Complete. RigL then reselected
to an empty active set, which is what finally made it visible.

Measured on the fixture: 4 of 4 expert modules match without the wrapper and 0
of 4 with it. The wrapper shares the module objects, so unwrapping changes which
names are looked up and nothing else.

These tests pin both halves: that the lookup survives the wrapper, and that a
mismatch raises instead of passing silently.
"""
import os

import pytest

E2E = "C:/Users/hm/Projects/e2e"
MODEL = "k8"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(E2E, MODEL)),
    reason="e2e fixture absent",
)


@pytest.fixture(scope="module")
def loaded():
    import torch.nn as nn
    from transformers import Qwen3MoeForCausalLM

    import usaf.model_factory as mf
    from usaf.train import _expert_module_names

    m = Qwen3MoeForCausalLM.from_pretrained(os.path.join(E2E, MODEL))
    cfg = mf.detect_model(os.path.join(E2E, MODEL))
    return m, nn.DataParallel(m), _expert_module_names(cfg)


def test_the_wrapper_breaks_the_plain_lookup(loaded):
    """The reason the helper exists, asserted so it cannot quietly become false."""
    _m, wrapped, names = loaded
    plain = [n for n, _ in wrapped.named_modules() if n in names]
    assert not plain, "DataParallel stopped prefixing names; revisit the helper"
    assert all(n.startswith("module.") for n, _ in wrapped.named_modules()
               if "experts" in n)


def test_the_helper_finds_everything_through_the_wrapper(loaded):
    from usaf.train import _expert_modules_by_name

    m, wrapped, names = loaded
    assert len(_expert_modules_by_name(m, names)) == len(names)
    assert len(_expert_modules_by_name(wrapped, names)) == len(names)


def test_the_wrapper_shares_the_module_objects(loaded):
    """Why unwrapping is safe rather than merely convenient."""
    from usaf.train import _expert_modules_by_name

    m, wrapped, names = loaded
    n0 = sorted(names)[0]
    assert _expert_modules_by_name(wrapped, names)[n0] is _expert_modules_by_name(m, names)[n0]


def test_a_mismatch_raises_instead_of_passing_silently(loaded):
    """The whole point: a missing expert must stop the run, not shrink it."""
    from usaf.train import _expert_modules_by_name

    m, _wrapped, names = loaded
    wrong = set(names)
    wrong.add("model.layers.99.mlp.experts")
    with pytest.raises(SystemExit, match="no expert modules found"):
        _expert_modules_by_name(m, wrong)
