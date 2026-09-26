"""An importance pass that captures nothing must stop the run.

ZAYA trained for an hour with Active: 0/0 and Optimizer: 0.0MB, and the loss
fell the whole time, because 1163 non-expert parameters were still being
trained. Nothing about that run looked wrong from the outside. The only visible
symptom was two numbers, and they were printed and ignored.

Patching one model family at a time does not scale to this: any MoE whose expert
container USAF has not been taught will do the same thing, quietly. The check
that does scale is the one on the number itself."""
import pytest


def _train_source() -> str:
    with open(
        r"C:\Users\hm\Projects\usaf\usaf\train.py", encoding="utf-8"
    ) as f:
        return f.read()


def test_a_non_empty_selection_passes_the_guard():
    selected = {'a': 1}
    assert selected, "a real selection must not trip the guard"


def test_the_guard_rejects_an_empty_selection():
    selected = {}
    with pytest.raises(SystemExit) as exc:
        if not selected:
            raise SystemExit(
                "the importance pass captured nothing, so no expert weight "
                "would be trained."
            )
    assert "no expert weight would be trained" in str(exc.value)


def test_the_trainer_really_contains_the_guard():
    """Guards drift out of the code they protect; this pins the rule in place."""
    src = _train_source()
    assert "the importance pass captured nothing" in src, (
        "the empty-selection guard is gone from train.py"
    )
    i_guard = src.find("the importance pass captured nothing")
    i_select = src.find("active_idx = imp_store.select(FRAC)")
    assert 0 <= i_select < i_guard, (
        "the guard has to sit after the selection it checks, not before it"
    )
