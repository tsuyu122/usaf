import pytest
import torch

from usaf.checkpoint import export_merged_weights
from usaf.quantization import dequantize_4bit, quantize_4bit


@pytest.fixture
def tiny_q4(tmp_path):
    """A small quantized checkpoint in the positional tuple form the writer
    actually produces, with two groups so a test can train inside one and
    leave the other completely alone.
    """
    torch.manual_seed(0)
    w = (torch.randn(256) * 0.05).half()
    path = tmp_path / "experts_q4.pt"
    torch.save({"layer.experts.gate_up_proj": quantize_4bit(w, group_size=128)}, path)
    # The reference is the *dequantized* original, not w: 4-bit quantization
    # is lossy, so comparing against w would fail even for a perfect export.
    deq = dequantize_4bit(*quantize_4bit(w, group_size=128)).reshape(-1)
    return str(path), deq


def _export(qpath, tmp_path, fname, masters, active, group_size=128):
    out = str(tmp_path / "out.pt")
    export_merged_weights(qpath, masters, active, out, group_size=group_size)
    return torch.load(out, map_location="cpu", weights_only=False)[fname]


def test_untouched_groups_survive_the_export_bit_for_bit(tiny_q4, tmp_path):
    """The README promised an export leaves everything outside the active set
    bit-identical. Requantizing a whole tensor recomputes every group scale,
    so untouched groups drifted even with nothing trained in them - on the
    4-layer fixture that moved 46777 of 770708 weights by up to 4.3e-04.
    """
    qpath, original = tiny_q4
    fname = "layer.experts.gate_up_proj"

    merged = _export(
        qpath, tmp_path, fname,
        {fname: torch.tensor([0.9], dtype=torch.float16)},
        {fname: torch.tensor([0], dtype=torch.long)},
    )
    new = dequantize_4bit(merged["q"], merged["s"], merged["z"], merged["shape"])
    new = new.reshape(-1)

    untouched = slice(128, 256)
    assert torch.equal(new[untouched], original[untouched]), (
        "a group with no trained weight came back different"
    )


def test_a_trained_weight_actually_reaches_the_export(tiny_q4, tmp_path):
    """The counterpart: preservation must not be so eager that it drops the
    training too.
    """
    qpath, original = tiny_q4
    fname = "layer.experts.gate_up_proj"

    merged = _export(
        qpath, tmp_path, fname,
        {fname: torch.tensor([0.9], dtype=torch.float16)},
        {fname: torch.tensor([0], dtype=torch.long)},
    )
    new = dequantize_4bit(merged["q"], merged["s"], merged["z"], merged["shape"])
    assert not torch.equal(new.reshape(-1)[0], original[0]), (
        "the trained value was dropped, so the export carries no training"
    )


def test_export_handles_the_dict_entry_form(tmp_path):
    """The same restoration has to work when the file uses {"q","s","z"}
    rather than the positional tuple. Skipping it for one of the two formats
    means half the checkpoints in the wild keep drifting.
    """
    torch.manual_seed(1)
    w = (torch.randn(256) * 0.05).half()
    q, s, z, shape = quantize_4bit(w, group_size=128)
    path = tmp_path / "d.pt"
    torch.save({"l": {"q": q, "s": s, "z": z, "shape": shape, "group_size": 128}}, path)

    original = dequantize_4bit(q, s, z, shape).reshape(-1)
    merged = _export(
        str(path), tmp_path, "l",
        {"l": torch.tensor([0.9], dtype=torch.float16)},
        {"l": torch.tensor([0], dtype=torch.long)},
    )
    new = dequantize_4bit(merged["q"], merged["s"], merged["z"], merged["shape"])
    assert torch.equal(new.reshape(-1)[128:], original[128:]), (
        "the dict entry form skipped the restoration"
    )
