import pytest

"""A selector that inverts its own percentile keeps the wrong end of the scores.

ThresholdSelector took the partition index to be the percentile itself, so the
fraction it kept was the mirror image of the one asked for: percentile=100
returned only the single highest score, and percentile=0 returned everything.
DynamicSelector relies on 99.98 to mean "keep almost everything", and got a
single weight instead - which would have left a RigL step training one expert
per tensor while appearing to select nearly all of them.
"""
import torch

from usaf.selector import DynamicSelector, ThresholdSelector, TopKSelector


@pytest.fixture
def scores():
    g = torch.Generator().manual_seed(0)
    return {"a": torch.rand(100, generator=g), "b": torch.rand(200, generator=g)}


def _count(masks):
    return int(sum(m.sum().item() for m in masks.values()))


@pytest.mark.parametrize("percentile,expected", [
    (1, 3),
    (5, 15),
    (25, 75),
    (50, 150),
    (75, 225),
])
def test_threshold_keeps_the_top_percentile(scores, percentile, expected):
    got = _count(ThresholdSelector(percentile).select(scores))
    assert abs(got - expected) <= 2, (
        f"percentile={percentile} kept {got}, expected about {expected}"
    )


def test_percentile_100_keeps_everything(scores):
    """The end that was broken: asking for all of it returned one element."""
    assert _count(ThresholdSelector(100).select(scores)) == 300


def test_percentile_0_keeps_almost_nothing(scores):
    got = _count(ThresholdSelector(0).select(scores))
    assert got <= 2, f"asking for none of it kept {got}"


def test_dynamic_selector_keeps_its_intended_fraction(scores):
    """DynamicSelector goes through ThresholdSelector(99.98). Before the fix
    this returned a single weight, so RigL would train one expert per tensor
    while looking like it selected nearly all of them.
    """
    d = DynamicSelector(50, 2)
    assert _count(d.update_mask(scores)) == 50


def test_topk_is_unaffected_by_the_percentile_fix(scores):
    for k, expected in ((1, 1), (50, 50), (300, 300), (9999, 300)):
        assert _count(TopKSelector(k).select(scores)) == expected
