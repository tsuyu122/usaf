"""A quantized file from the wrong model must not train quietly.

The expert parameters are cleared and refilled from the q4 cache, and the
training shapes are read back out of the same file, so nothing ever compared
it against the model. Pointing tiny-moe at the Mixtral experts_q4.pt produced a
plausible loss curve, a green run, and an entire model running on the wrong
weights. The shapes were right there in the file: (4, 256, 512) against a
model whose experts are (4, 64, 512).
"""
import pytest
import torch

from usaf.train import _expert_shapes, _validate_quant_weights


def _entry(shape):
    return (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.Size(shape))


def test_matching_file_is_accepted():
    expected = {
        "model.layers.0.mlp.experts.gate_up_proj": (4, 64, 512),
        "model.layers.0.mlp.experts.down_proj": (4, 512, 32),
    }
    q_dict = {k: _entry(v) for k, v in expected.items()}
    _validate_quant_weights(q_dict, expected, "experts_q4.pt")


def test_dict_style_entries_are_accepted_too():
    expected = {"w": (2, 3, 4)}
    q_dict = {"w": {"q": torch.zeros(1), "s": torch.zeros(1),
                   "z": torch.zeros(1), "shape": (2, 3, 4)}}
    _validate_quant_weights(q_dict, expected, "experts_q4.pt")


def test_wrong_intermediate_size_is_refused():
    expected = {"model.layers.0.mlp.experts.gate_up_proj": (4, 64, 512)}
    q_dict = {"model.layers.0.mlp.experts.gate_up_proj": _entry((4, 256, 512))}
    with pytest.raises(SystemExit) as ei:
        _validate_quant_weights(q_dict, expected, "experts_q4.pt")
    msg = str(ei.value)
    assert "do not match" in msg
    assert "(4, 256, 512)" in msg and "(4, 64, 512)" in msg


def test_missing_tensor_is_refused():
    expected = {"model.layers.0.mlp.experts.gate_up_proj": (4, 64, 512)}
    with pytest.raises(SystemExit) as ei:
        _validate_quant_weights({}, expected, "experts_q4.pt")
    assert "not in the quantized file" in str(ei.value)


def test_extra_tensors_in_the_file_are_not_a_problem():
    """A file may carry layers this run is not training; that is not a mismatch."""
    expected = {"a": (2, 2)}
    q_dict = {"a": _entry((2, 2)), "b": _entry((9, 9))}
    _validate_quant_weights(q_dict, expected, "experts_q4.pt")


def test_expert_shapes_reads_the_model_before_the_cache_clears_it():
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.zeros(4, 64, 512))
            self.down_proj = torch.nn.Parameter(torch.zeros(4, 512, 32))

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    net = Net()
    got = _expert_shapes(net, {"layers.0.experts"})
    assert got == {
        "layers.0.experts.gate_up_proj": (4, 64, 512),
        "layers.0.experts.down_proj": (4, 512, 32),
    }


def test_validation_catches_a_real_mismatched_fixture(tmp_path):
    """Against the actual fixture files, not a hand-written stand-in."""
    import os

    e2e = r"C:\Users\hm\Projects\e2e"
    moe_q4 = os.path.join(e2e, "tiny-moe-q4", "experts_q4.pt")
    mixtral_q4 = os.path.join(e2e, "tiny-mixtral-q4", "experts_q4.pt")
    if not (os.path.exists(moe_q4) and os.path.exists(mixtral_q4)):
        pytest.skip("e2e fixtures not present")

    good = torch.load(moe_q4, map_location="cpu", weights_only=False)
    wrong = torch.load(mixtral_q4, map_location="cpu", weights_only=False)
    expected = {k: tuple(v[3]) for k, v in good.items()}

    _validate_quant_weights(good, expected, moe_q4)
    with pytest.raises(SystemExit):
        _validate_quant_weights(wrong, expected, mixtral_q4)
