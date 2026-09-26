"""Why one trainer broke on multi-GPU and the other did not.

usaf/train.py installs its capture hooks by matching module names against an
exact set built from the detected expert prefix, so DataParallel's "module."
prefix made every comparison miss and no hook was installed. The root
train.py matches with endswith(".mlp.experts") instead, and the prefix sits at
the front of the name, so the suffix still matches and its hooks installed
correctly.

That asymmetry is the whole reason the defect hid: the same command, the same
wrapper, the same run, one entrypoint dead and one alive, and nothing on either
path said so. These tests pin the asymmetry so it stays visible.

The root train.py is a script that trains a 30B model, so it is not imported.
The suffix check is exercised against real wrappers instead.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOT_SCRIPT = ROOT / "train.py"
USAF = ROOT / "usaf" / "train.py"
E2E = pathlib.Path("C:/Users/hm/Projects/e2e")
FIXTURE = E2E / "k8"

pytestmark = pytest.mark.skipif(
    not ROOT_SCRIPT.is_file() or not USAF.is_file(), reason="trainers not present"
)


def test_the_two_trainers_discover_experts_differently():
    a = ROOT_SCRIPT.read_text(encoding="utf-8")
    b = USAF.read_text(encoding="utf-8")
    assert 'endswith(".mlp.experts")' in a, "the root script matches by suffix"
    assert "_expert_modules" in b and "in _expert_modules" in b, (
        "usaf/train.py matches against an exact set"
    )


def test_the_suffix_match_survives_a_wrapper():
    """The property that kept the root script working, on a real wrapper."""
    if not FIXTURE.is_dir():
        pytest.skip("e2e fixture absent")
    import torch.nn as nn
    from transformers import Qwen3MoeForCausalLM

    m = Qwen3MoeForCausalLM.from_pretrained(str(FIXTURE))
    plain = [n for n, _ in m.named_modules() if n.endswith(".mlp.experts")]
    wrapped = [
        n for n, _ in nn.DataParallel(m).named_modules()
        if n.endswith(".mlp.experts")
    ]
    assert len(plain) == 4, plain
    assert len(wrapped) == len(plain), (
        f"the suffix match stopped surviving the wrapper: {plain} vs {wrapped}"
    )
    assert all(n.startswith("module.") for n in wrapped), (
        "DataParallel stopped prefixing names; this test no longer proves the "
        "asymmetry it was written for"
    )


def test_the_exact_match_does_not_survive_a_wrapper():
    """The defect, pinned as a fact about the mechanism rather than a bug."""
    if not FIXTURE.is_dir():
        pytest.skip("e2e fixture absent")
    import torch.nn as nn
    from transformers import Qwen3MoeForCausalLM

    m = Qwen3MoeForCausalLM.from_pretrained(str(FIXTURE))
    expected = {
        n for n, _ in m.named_modules() if n.endswith(".mlp.experts")
    }
    wrapped = nn.DataParallel(m)
    exact_hits = [n for n, _ in wrapped.named_modules() if n in expected]
    assert not exact_hits, (
        "the exact set now matches the wrapper, so unwrapping is unnecessary; "
        f"revisit _expert_modules_by_name: {exact_hits}"
    )


def test_usaf_unwraps_so_the_exact_set_still_matches():
    from usaf.train import _expert_modules_by_name

    if not FIXTURE.is_dir():
        pytest.skip("e2e fixture absent")
    import torch.nn as nn
    from transformers import Qwen3MoeForCausalLM

    m = Qwen3MoeForCausalLM.from_pretrained(str(FIXTURE))
    expected = {
        n for n, _ in m.named_modules() if n.endswith(".mlp.experts")
    }
    wrapped = nn.DataParallel(m)
    assert set(_expert_modules_by_name(wrapped, expected)) == expected
    assert set(_expert_modules_by_name(m, expected)) == expected
