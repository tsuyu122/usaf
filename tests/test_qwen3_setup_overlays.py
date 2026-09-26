"""The checkpoint half that does not need DirectML, which will not load here.

load_qwen3_streaming needs torch_directml_native, so it cannot be exercised on
this machine at all - which is why the RoPE bug that lived in it survived a fix
that had already landed in the other copy. apply_checkpoint_overlays is plain
torch and is the half that decides whether a loaded model is the trained one.
"""
import pytest
import torch

from usaf.qwen3_setup import apply_checkpoint_overlays


class _Cache:
    def __init__(self):
        self.overlays = {}


def _write(tmp_path, active, masters):
    p = tmp_path / "ck.pt"
    torch.save({"active_idx": active, "masters": masters}, p)
    return str(p)


def test_a_well_formed_checkpoint_installs_one_overlay_per_tensor(tmp_path):
    active = {"model.layers.0.mlp.experts.gate_up_proj": torch.tensor([1, 5, 9])}
    masters = {"model.layers.0.mlp.experts.gate_up_proj": torch.tensor([1.0, 2.0, 3.0])}
    c = _Cache()
    n = apply_checkpoint_overlays(c, _write(tmp_path, active, masters))
    assert n == 1
    idx, vals = c.overlays["model.layers.0.mlp.experts.gate_up_proj"]
    assert idx.tolist() == [1, 5, 9]
    assert torch.allclose(vals.reshape(-1), torch.tensor([1.0, 2.0, 3.0]))


def test_a_master_shorter_than_its_index_list_is_refused(tmp_path):
    # The failure this guards: a scatter from a master that does not match its
    # indices either throws from inside a forward or, worse, writes the wrong
    # trained values over the right positions - a model that has quietly been
    # loaded with somebody else parameters.
    active = {"t": torch.tensor([1, 5, 9])}
    masters = {"t": torch.tensor([1.0, 2.0])}
    with pytest.raises(ValueError, match="active_idx has 3 entries"):
        apply_checkpoint_overlays(_Cache(), _write(tmp_path, active, masters))


def test_a_tensor_with_indices_and_no_master_is_refused(tmp_path):
    active = {"t": torch.tensor([1, 2])}
    masters = {"other": torch.tensor([1.0, 2.0])}
    with pytest.raises(KeyError, match="carries no master"):
        apply_checkpoint_overlays(_Cache(), _write(tmp_path, active, masters))


def test_a_refused_checkpoint_leaves_nothing_installed(tmp_path):
    # Installing as it goes leaves a half-applied checkpoint behind when a later
    # tensor fails, and a caller that catches the error and carries on then runs
    # a model built from a mix of trained and original weights - the same quiet
    # wrongness the length check exists to prevent, one tensor later.
    active = {"good": torch.tensor([1, 2]), "bad": torch.tensor([1, 2, 3])}
    masters = {"good": torch.tensor([1.0, 2.0]), "bad": torch.tensor([1.0])}
    c = _Cache()
    with pytest.raises(ValueError):
        apply_checkpoint_overlays(c, _write(tmp_path, active, masters))
    assert c.overlays == {}, c.overlays

