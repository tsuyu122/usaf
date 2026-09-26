"""The memory budget was computed and then thrown away.

--train-from defaults to 0 and its help says 0=auto. main() always passes that
parsed 0 to get_trainable_layers, which read it as a literal start and returned
range(0, num_layers) - every layer, on every model, at every size.

_auto_configure_training goes to real trouble to work out how many layers fit in
VRAM and host RAM. It set max_trainable_layers to 14 for ZAYA1-8B on a 15.6 GB
Tesla T4. Nobody read it. A Kaggle run announced "Trainable: 40 layers (0-39)"
for an 8.84B model on a card that cannot hold them, and the sizing comment right
above it - which explains at length why the estimate must come from the card and
not the host - describes a number that never took effect.
"""
import pytest

from usaf.model_factory import MoEConfig, get_trainable_layers


def _cfg(n_layers=40, cap=14):
    c = MoEConfig()
    c.num_layers = n_layers
    c.max_trainable_layers = cap
    c.train_from = max(0, n_layers - cap)
    return c


def test_the_default_invocation_honours_the_budget():
    c = _cfg(40, 14)
    got = get_trainable_layers(c, 0)
    assert len(got) == 14, got
    assert got == set(range(26, 40)), got


def test_a_budget_of_zero_means_no_cap_not_one_layer():
    """An uncomputed budget must not silently shrink the run to one layer."""
    c = _cfg(40, 0)
    c.train_from = 0
    assert get_trainable_layers(c, 0) == set(range(40))


def test_an_explicit_train_from_still_wins():
    c = _cfg(40, 14)
    got = get_trainable_layers(c, 10)
    assert got == set(range(10, 40)), got


def test_none_means_auto_too():
    c = _cfg(40, 14)
    assert get_trainable_layers(c, None) == set(range(26, 40))


def test_a_cap_larger_than_the_model_is_all_of_it():
    c = _cfg(8, 14)
    assert get_trainable_layers(c, 0) == set(range(8))


def test_the_old_behaviour_is_the_bug_not_the_contract():
    """Pin what the default used to produce, so the fix cannot be undone.

    Every layer is exactly what the run did on Kaggle, and it is exactly what
    this function must no longer return.
    """
    c = _cfg(40, 14)
    buggy = set(range(0, c.num_layers))
    fixed = get_trainable_layers(c, 0)
    assert buggy != fixed
    assert len(buggy) == 40
    assert len(fixed) == 14


def test_a_train_from_past_the_end_is_still_refused():
    c = _cfg(40, 14)
    with pytest.raises(SystemExit):
        get_trainable_layers(c, 40)

