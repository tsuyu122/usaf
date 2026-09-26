"""Overlays carry trained weights into a quantized expert cache.

An overlay is a pair of flat indices into the full tensor and the compact
values for those positions. If the scatter lands on a copy instead of the
tensor that is kept, the model quietly serves pretrained weights for a run
that believes it fine-tuned - and the loss curve still looks like training.
"""
import torch

from usaf.moe_loader import _apply_overlay


def test_overlay_lands_on_a_contiguous_tensor():
    t = torch.zeros(4, 6)
    idx = torch.tensor([0, 5, 11, 23])
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = _apply_overlay(t, idx, vals)
    assert out.reshape(-1)[[0, 5, 11, 23]].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert int(t.abs().sum()) == 10


def test_overlay_lands_on_a_non_contiguous_slice():
    """reshape(-1) copies on a narrow slice, so the scatter would vanish.

    This is the case that makes the difference between reshape and view.
    """
    wide = torch.zeros(8, 4)
    t = wide[:, :2]
    assert not t.is_contiguous()

    idx = torch.tensor([0, 2, 4, 6])
    vals = torch.tensor([5.0, 6.0, 7.0, 8.0])
    out = _apply_overlay(t, idx, vals)

    assert out.reshape(-1)[[0, 2, 4, 6]].tolist() == [5.0, 6.0, 7.0, 8.0], (
        "the overlay was scattered into a copy and did nothing"
    )


def test_overlay_casts_values_to_the_tensor_dtype():
    t = torch.zeros(8, dtype=torch.float16)
    out = _apply_overlay(t, torch.tensor([0, 1]), torch.tensor([1.5, 2.5], dtype=torch.float32))
    assert out.dtype == torch.float16
    assert out[:2].tolist() == [1.5, 2.5]
    assert out[2:].abs().sum().item() == 0.0


def test_overlay_does_not_touch_positions_it_was_not_given():
    t = torch.zeros(4, 4)
    out = _apply_overlay(t, torch.tensor([1, 7]), torch.tensor([9.0, 9.0]))
    got = out.reshape(-1).tolist()
    assert got[1] == 9.0 and got[7] == 9.0
    assert sum(1 for v in got if v == 0.0) == 14
