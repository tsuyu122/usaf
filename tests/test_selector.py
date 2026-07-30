"""Tests for weight selectors (TopK, Threshold, Dynamic)."""
import torch

from usaf.selector import TopKSelector, ThresholdSelector, DynamicSelector


def test_topk_selector_empty():
    sel = TopKSelector(k=100)
    result = sel.select({})
    assert result == {}


def test_topk_selector_basic():
    scores = {
        "layer1.weight": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        "layer2.weight": torch.tensor([0.1, 0.2, 0.3]),
    }
    sel = TopKSelector(k=3)
    result = sel.select(scores)
    active_count = sum(m.sum().item() for m in result.values())
    assert active_count == 3


def test_topk_selector_all_active():
    scores = {"w1": torch.tensor([1.0, 2.0, 3.0])}
    sel = TopKSelector(k=10)
    result = sel.select(scores)
    assert result["w1"].sum().item() == 3


def test_threshold_selector_basic():
    scores = {"w1": torch.tensor([0.1, 0.5, 0.9, 1.0])}
    sel = ThresholdSelector(percentile=50.0)
    result = sel.select(scores)
    active = result["w1"]
    assert active.sum().item() == 2
    assert active[2].item() is True
    assert active[3].item() is True


def test_dynamic_selector_topk():
    sel = DynamicSelector(initial_k=100, reselect_every_n_steps=50, selection="topk")
    assert sel.should_reselect() is False  # step 1
    for _ in range(48):
        sel.should_reselect()
    assert sel.should_reselect() is True  # step 50
    assert sel.should_reselect() is False  # step 51


def test_dynamic_selector_update_mask():
    sel = DynamicSelector(initial_k=2, reselect_every_n_steps=10)
    scores = {"w1": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])}
    mask = sel.update_mask(scores, k=2)
    assert "w1" in mask
    assert mask["w1"].sum().item() == 2


def test_dynamic_selector_active_mask_property():
    sel = DynamicSelector(initial_k=10, reselect_every_n_steps=10)
    assert sel.active_mask == {}
    scores = {"w1": torch.tensor([1.0, 2.0])}
    sel.update_mask(scores, k=2)
    assert len(sel.active_mask) == 1
