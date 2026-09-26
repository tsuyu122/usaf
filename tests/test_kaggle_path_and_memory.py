"""Two bugs the first Kaggle run surfaced, both about paths and memory.

Neither would show up on the machine the project was developed on.
"""
import json
import os
import subprocess
import sys

import pytest

E2E = "C:/Users/hm/Projects/e2e"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(E2E, "tiny-moe")),
    reason="e2e fixtures absent",
)


def test_the_quantizer_and_the_trainer_agree_about_where_q4_lives(tmp_path):
    """usaf.quantize writes next to the model; the trainer looked next to the cwd.

    The two only agreed when the model was a directory sitting in the current
    directory. Any other path - a subdirectory, an absolute path, or what the
    Kaggle kernel does - quantized fine and then failed to find the file, which
    is the second half of the ordinary workflow reporting that the first half
    did not happen.
    """
    sub = tmp_path / "sub"
    model = sub / "m"
    sub.mkdir(parents=True)

    import shutil

    shutil.copytree(os.path.join(E2E, "tiny-moe"), model)

    q = subprocess.run(
        [sys.executable, "-m", "usaf.quantize", "--model", str(model)],
        cwd=E2E,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = (q.stdout or "") + (q.stderr or "")
    assert "Traceback" not in log, log[-600:]
    produced = sub / "m-q4" / "experts_q4.pt"
    assert produced.exists(), log[-400:]

    # A forward-slash path, on purpose.
    #
    # The call site used to strip the directory with split("/")[-1]. That is a
    # no-op on Windows, where the separator is a backslash, and strips the
    # directory on Linux - so a backslash path could not tell the broken code
    # from the fixed code, and this test passed against the broken one. A
    # forward-slash path fails here exactly the way it fails on Kaggle.
    rel = os.path.relpath(model, E2E).replace(os.sep, "/")
    assert "/" in rel, rel

    t = subprocess.run(
        [
            sys.executable,
            "-m",
            "usaf.train",
            "--model",
            rel,
            "--dataset",
            "data.jsonl",
            "--seq-len",
            "32",
            "--microbatch",
            "2",
            "--frac",
            "0.1",
            "--train-from",
            "0",
            "--steps",
            "2",
            "--eval-every",
            "0",
            "--save-every",
            "0",
            "--checkpoint-dir",
            str(tmp_path),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        cwd=E2E,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (t.stdout or "") + (t.stderr or "")
    assert "not found" not in out, out[-600:]
    assert "Q4 weights" in out, out[-400:]
    assert "Loss:" in out, out[-400:]


ZAYA = {
    "model_type": "zaya",
    "hidden_size": 2048,
    "intermediate_size": 2048,
    "moe_intermediate_size": 512,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 262272,
    "max_position_embeddings": 4096,
    "num_experts": 16,
    "num_experts_per_tok": 1,
    "num_local_experts": 16,
    "decoder_sparse_step": 1,
    "norm_topk_prob": True,
    "tie_word_embeddings": True,
    "partial_rotary_factor": 0.5,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
}


@pytest.fixture(scope="module")
def zaya_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("zaya") / "z"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(ZAYA), encoding="utf-8")
    return str(d)


def test_detect_model_knows_zaya(zaya_dir):
    """The final run is ZAYA1-8B, so the config has to resolve without help."""
    from usaf.model_factory import detect_model

    c = detect_model(zaya_dir)
    assert c.is_moe
    assert c.num_layers == 40
    assert c.num_experts == 16
    assert c.num_experts_per_tok == 1
    assert c.expert_intermediate == 512
    assert c.expert_param_names == ["gate_up_proj", "down_proj"]
    assert c.expert_prefix == "model.layers.{i}.mlp.experts"


def test_the_layer_budget_follows_the_card_not_the_host(zaya_dir):
    """_auto_configure_training took vram_gb, stored it, and read it nowhere.

    On a Kaggle kernel, roughly 30 GB of host RAM and 15.6 GB of VRAM, this
    sized ZAYA1-8B at 29 trainable layers, which is 17.9 GB of layers in a
    15.6 GB card. Doubling the VRAM to 79 GB changed the answer not at all.
    """
    from usaf.model_factory import detect_model

    small = detect_model(zaya_dir, vram_gb=15.6, system_ram_gb=200.0)
    large = detect_model(zaya_dir, vram_gb=79.0, system_ram_gb=200.0)

    assert small.max_trainable_layers < large.max_trainable_layers, (
        "a bigger card produced the same number of trainable layers"
    )

    for c in (small, large):
        vram = c.estimated_vram_gb
        budget = c.max_trainable_layers * c.estimated_per_layer_gb
        assert budget <= vram, (
            f"the layers alone need {budget:.1f} GB in a {vram:.1f} GB card"
        )


def test_host_ram_still_bounds_it_when_it_is_the_smaller(zaya_dir):
    """The two bounds are a minimum, not a replacement."""
    from usaf.model_factory import detect_model

    a = detect_model(zaya_dir, vram_gb=79.0, system_ram_gb=30.0)
    b = detect_model(zaya_dir, vram_gb=79.0, system_ram_gb=200.0)

    assert a.max_trainable_layers < b.max_trainable_layers
