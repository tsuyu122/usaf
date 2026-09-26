"""The CLI's failure modes must be readable, not incidental.

Every one of these used to fail somewhere unrelated to the actual mistake: a
missing dataset became an empty list, which became "ZeroDivisionError: integer
modulo by zero" from the progress bar hundreds of lines later; a train-from past
the end of the model became "min() arg is an empty sequence"; a mistyped model
path came back from HuggingFace as "Repo id must use alphanumeric chars", which
never mentions the path.
"""
import json

import pytest

from usaf.model_factory import MoEConfig, get_trainable_layers
from usaf.train import _load_dataset


def _cfg(num_layers: int = 4) -> MoEConfig:
    return MoEConfig(
        model_path="x",
        num_layers=num_layers,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
    )


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


def test_missing_dataset_says_so(tmp_path):
    missing = str(tmp_path / "nope.jsonl")
    with pytest.raises(SystemExit, match="dataset not found"):
        _load_dataset(missing, 32)


def test_malformed_json_is_reported_not_swallowed(tmp_path, capsys):
    p = _write(tmp_path, "bad.jsonl", ["this is not json at all"])
    with pytest.raises(SystemExit, match="no usable sample"):
        _load_dataset(p, 32)
    out = capsys.readouterr().out
    assert "not valid JSON" in out, out
    assert "bad.jsonl:1" in out, "the diagnostic must name the offending line"


def test_non_list_input_ids_is_reported(tmp_path, capsys):
    p = _write(tmp_path, "bad.jsonl", [json.dumps({"input_ids": "nope", "labels": 3})])
    with pytest.raises(SystemExit, match="no usable sample"):
        _load_dataset(p, 32)
    assert "input_ids is not a list" in capsys.readouterr().out


def test_wrong_seq_len_is_reported(tmp_path):
    p = _write(tmp_path, "short.jsonl", [json.dumps({"input_ids": [1, 2, 3], "labels": [1, 2, 3]})])
    with pytest.raises(SystemExit, match="no usable sample"):
        _load_dataset(p, 32)


def test_empty_file_is_reported(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="no usable sample"):
        _load_dataset(str(p), 32)


def test_good_dataset_still_loads_and_splits(tmp_path):
    rows = [
        json.dumps({"input_ids": [i % 100] * 32, "labels": [i % 100] * 32})
        for i in range(40)
    ]
    p = _write(tmp_path, "ok.jsonl", rows)
    train, ev, held = _load_dataset(p, 32)
    assert len(train) == 20
    assert len(ev) == 10
    assert len(held) == 10
    assert all(len(s["input_ids"]) == 32 for s in train)


def test_train_from_past_the_end_is_rejected():
    with pytest.raises(SystemExit, match="past the last layer"):
        get_trainable_layers(_cfg(4), 99)


def test_train_from_negative_is_rejected():
    with pytest.raises(SystemExit, match="must not be negative"):
        get_trainable_layers(_cfg(4), -1)


def test_train_from_last_layer_is_allowed():
    # Starting at the final layer trains exactly one layer, which is legal.
    assert get_trainable_layers(_cfg(4), 3) == {3}


def test_train_from_zero_trains_everything():
    assert get_trainable_layers(_cfg(4), 0) == {0, 1, 2, 3}
