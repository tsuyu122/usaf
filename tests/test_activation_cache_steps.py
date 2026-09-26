"""The activation cache indexes by step, not by position in a list.

The cache stored one list per module and get_cached indexed that list by the
step number. Those are the same thing only while every module is called exactly
once per step. A module called twice - a reused decoder layer, a second pass,
anything that wraps the block - shifts every later index by one, and the cache
then returns the previous step activation with no error at all.
"""
import pytest
import torch
import torch.nn as nn

from usaf.cache import ActivationCache


class Block(nn.Module):
    """Named Block, which is what the hook looks for in the class name.

    It adds a marker instead of computing anything, so the value that comes back
    out of the cache says exactly which call produced it.
    """

    def __init__(self):
        super().__init__()
        self.marker = 0.0

    def forward(self, x):
        return x + self.marker


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def forward(self, x, times=1):
        for _ in range(times):
            x = self.block(x)
        return x


@pytest.fixture()
def wired():
    m = Model()
    c = ActivationCache()
    c.register_hooks(m)
    yield c, m
    c.remove_hooks()


def _step(m, c, value, times=1):
    m.block.marker = float(value)
    m(torch.zeros(2, 4), times=times)
    c.advance_step()


def test_one_call_per_step_returns_that_step(wired):
    c, m = wired
    for step, value in enumerate([1.0, 2.0, 3.0]):
        _step(m, c, value)
    for step, value in enumerate([1.0, 2.0, 3.0]):
        got = c.get_cached("block", step)
        assert got is not None, f"nothing captured for step {step}"
        assert torch.allclose(got, torch.full((2, 4), value)), (
            f"step {step} returned another step", got.flatten()[0].item())


def test_a_module_called_twice_does_not_shift_later_steps(wired):
    # The failure this guards: step 0 fills indices 0 and 1, so step 1 lands on
    # index 2 and get_cached(name, 1) hands back step 0 second call.
    _step(m_ := wired[1], c := wired[0], 10.0, times=2)
    _step(m_, c, 20.0, times=2)
    _step(m_, c, 30.0, times=1)
    for step, value in enumerate([10.0, 20.0, 30.0]):
        got = c.get_cached("block", step)
        assert got is not None, f"step {step} returned nothing"
        assert torch.allclose(got, torch.full((2, 4), value)), (
            f"step {step} looks like a different step", got.flatten()[0].item())


def test_the_number_of_calls_in_a_step_is_visible(wired):
    c, m = wired
    _step(m, c, 1.0, times=3)
    assert c.calls_in_step("block", 0) == 3
    assert c.calls_in_step("block", 5) == 0


def test_a_step_with_no_capture_returns_none_rather_than_the_nearest(wired):
    c, m = wired
    _step(m, c, 1.0)
    assert c.get_cached("block", 0) is not None
    assert c.get_cached("block", 1) is None
    assert c.get_cached("block", 99) is None


def test_a_negative_step_is_rejected_instead_of_reading_the_newest(wired):
    c, m = wired
    _step(m, c, 1.0)
    _step(m, c, 2.0)
    with pytest.raises(ValueError, match="negative"):
        c.get_cached("block", -1)


def test_reset_clears_both_the_capture_and_the_invalidations(wired):
    c, m = wired
    _step(m, c, 1.0)
    assert c.get_cached("block", 0) is not None
    c.reset()
    assert c.get_cached("block", 0) is None
    assert c.cached_modules == set(), c.cached_modules


def test_invalidate_stops_capture_for_that_module(wired):
    c, m = wired
    c.invalidate({'block'})
    _step(m, c, 1.0)
    assert c.get_cached('block', 0) is None
    assert c.cached_modules == set()

