"""Tests for checkpoint save/load/export."""

import os
import tempfile
import torch
from usaf.checkpoint import (
    save_sparse_checkpoint,
    load_sparse_checkpoint,
    export_merged_weights,
    get_checkpoint_metadata,
)


def test_save_load_roundtrip():
    masters = {
        "w1": torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0])),
        "w2": torch.nn.Parameter(torch.tensor([4.0, 5.0])),
    }
    active_idx = {
        "w1": torch.tensor([0, 10, 20], dtype=torch.long),
        "w2": torch.tensor([5, 15], dtype=torch.long),
    }
    opt_state = {"step": 3, "m": {"w1": torch.zeros(3)}, "v": {"w1": torch.zeros(3)}}
    config = {
        "model": "test-model",
        "steps": 100,
        "frac": 0.005,
        "lr": 2e-4,
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        save_sparse_checkpoint(
            path, masters, active_idx, opt_state,
            config, step=5, losses=[2.5, 2.3, 2.1],
            train_layers=[11, 12, 13],
            metric=2.1,
        )

        loaded = load_sparse_checkpoint(path)
        assert loaded["format"] == "usaf-sparse-v1"
        assert loaded["step"] == 5
        assert len(loaded["losses"]) == 3
        assert loaded["metric"] == 2.1
        assert "w1" in loaded["active_idx"]
        assert "w1" in loaded["masters"]
        assert torch.equal(loaded["active_idx"]["w1"], active_idx["w1"])
        assert loaded["config"]["model"] == "test-model"
        assert loaded["train_layers"] == [11, 12, 13]


def test_metadata():
    masters = {"w": torch.nn.Parameter(torch.tensor([1.0, 2.0]))}
    active_idx = {"w": torch.tensor([0, 5], dtype=torch.long)}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        save_sparse_checkpoint(
            path, masters, active_idx, {},
            {"model": "test"}, step=10, losses=[1.0],
            train_layers=[0],
        )
        meta = get_checkpoint_metadata(path)
        assert meta["step"] == 10
        assert meta["n_active"] == 2
        assert meta["n_masters"] == 1
        assert meta["format"] == "usaf-sparse-v1"


def test_save_overwrites_existing():
    masters = {"w": torch.nn.Parameter(torch.tensor([1.0]))}
    active_idx = {"w": torch.tensor([0], dtype=torch.long)}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        save_sparse_checkpoint(path, masters, active_idx, {}, {}, step=1, losses=[], train_layers=[0])
        save_sparse_checkpoint(path, masters, active_idx, {}, {}, step=2, losses=[], train_layers=[0])
        loaded = load_sparse_checkpoint(path)
        assert loaded["step"] == 2


def test_load_invalid_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.pt")
        torch.save({"format": "unknown"}, path)
        try:
            load_sparse_checkpoint(path)
            assert False, "Should have raised"
        except ValueError:
            pass


def test_export_merged_weights(tmp_path):
    from usaf.quantization import quantize_state_dict

    orig_weights = {
        "expert.w1": torch.randn(8, 8, dtype=torch.float16),
        "expert.w2": torch.randn(4, 4, dtype=torch.float16),
    }
    qpath = str(tmp_path / "orig_q4.pt")
    qdict = quantize_state_dict(orig_weights, group_size=4)
    torch.save(qdict, qpath)

    masters = {
        "expert.w1": torch.tensor([99.0, 99.0]),
        "expert.w2": torch.tensor([88.0]),
    }
    active_idx = {
        "expert.w1": torch.tensor([0, 1], dtype=torch.long),
        "expert.w2": torch.tensor([3], dtype=torch.long),
    }

    outpath = str(tmp_path / "merged_q4.pt")
    result = export_merged_weights(qpath, masters, active_idx, outpath, group_size=4)
    assert os.path.exists(result)

    merged = torch.load(result, weights_only=True)
    from usaf.quantization import dequantize_4bit

    for fname in orig_weights:
        entry = merged[fname]
        if isinstance(entry, dict) and "q" in entry:
            t = dequantize_4bit(
                entry["q"], entry["s"], entry["z"], entry["shape"],
                group_size=entry.get("group_size", 4)
            )
        else:
            t = entry.clone()

        if fname in masters:
            aidx = active_idx[fname]
            original = orig_weights[fname].reshape(-1).clone()
            trained = masters[fname]

            for i, idx in enumerate(aidx):
                expected_val = trained[i].item()
                assert abs(t.reshape(-1)[idx].item() - expected_val) < 0.1,                     f"{fname}[{idx}] expected {expected_val}, got {t.reshape(-1)[idx].item()}"
            for i in range(t.reshape(-1).numel()):
                if i not in aidx:
                    original_val = original[i].item()
                    assert abs(t.reshape(-1)[i].item() - original_val) < 0.1
