import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from install_lmstudio import install, main, survey  # noqa: E402


@pytest.fixture()
def good(tmp_path):
    """A model directory that is actually loadable, not one that looks it."""
    d = tmp_path / "export"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "architectures": ["GraniteMoeForCausalLM"],
        "model_type": "granitemoe",
    }))
    save_file({"a": torch.zeros(2, 2), "b": torch.zeros(2)}, str(d / "model.safetensors"))
    (d / "tokenizer.json").write_text("{}")
    return d


def test_a_good_export_passes_and_counts_its_tensors(good):
    info = survey(good)
    assert info["ok"], info["problems"]
    assert info["n_tensors"] == 2
    assert info["architectures"] == ["GraniteMoeForCausalLM"]


def test_everything_wrong_is_reported_at_once(tmp_path):
    # One run to fix, not three: a directory that is short a config, a set of
    # weights and a tokenizer has to say all three, or fixing the first one
    # just reveals the second one on the next run.
    d = tmp_path / "vazio"
    d.mkdir()
    info = survey(d)
    assert not info["ok"]
    joined = " ".join(info["problems"])
    assert "config.json" in joined
    assert "pesos" in joined
    assert "tokenizer" in joined


def test_weights_that_do_not_open_are_reported(tmp_path):
    d = tmp_path / "ruim"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["X"]}))
    (d / "tokenizer.json").write_text("{}")
    (d / "model.safetensors").write_text("x")
    info = survey(d)
    assert not info["ok"]
    assert any("nao abrem" in x for x in info["problems"]), info["problems"]


def test_a_directory_that_is_not_there(tmp_path):
    info = survey(tmp_path / "nao-existe")
    assert not info["ok"]
    assert "diretorio" in " ".join(info["problems"])


def test_install_copies_and_the_copy_is_checked(good, tmp_path):
    root = tmp_path / "models"
    dest = install(good, "meu-modelo", root, "usaf")
    assert dest == root / "usaf" / "meu-modelo"
    assert (dest / "model.safetensors").is_file()
    assert survey(dest)["n_tensors"] == 2


def test_install_refuses_to_overwrite_without_force(good, tmp_path):
    root = tmp_path / "models"
    install(good, "meu-modelo", root, "usaf")
    with pytest.raises(SystemExit, match="--force"):
        install(good, "meu-modelo", root, "usaf")


def test_install_refuses_a_source_that_is_not_loadable(tmp_path):
    d = tmp_path / "vazio"
    d.mkdir()
    with pytest.raises(SystemExit, match="origem"):
        install(d, "meu-modelo", tmp_path / "models", "usaf")


def test_main_prints_and_returns_zero(good, tmp_path, capsys):
    rc = main([str(good), "meu-modelo", "--models-root", str(tmp_path / "m")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "instalado:" in out
    assert "2 tensores" in out


def test_main_returns_one_and_says_why(tmp_path, capsys):
    d = tmp_path / "vazio"
    d.mkdir()
    assert main([str(d), "meu-modelo", "--models-root", str(tmp_path / "m")]) == 1
    assert "nao serve" in capsys.readouterr().err

def test_force_replaces_what_was_there(good, tmp_path):
    # The refusal test above passes just as well if the guard always refuses, and
    # this one is what catches that. A guard that can never be talked out of it is
    # a guard that stops the next install, not a guard.
    root = tmp_path / "models"
    install(good, "meu-modelo", root, "usaf")
    stale = root / "usaf" / "meu-modelo" / "model.safetensors"
    stale.unlink()
    stale.write_bytes(b"velho")
    dest = install(good, "meu-modelo", root, "usaf", force=True)
    assert survey(dest)["n_tensors"] == 2
    assert not (dest / "model.safetensors").read_bytes().startswith(b"velho")
