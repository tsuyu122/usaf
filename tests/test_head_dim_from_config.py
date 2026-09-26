"""A config that declares head_dim is the authority, and the head count has to
follow from it rather than from a division.

ZAYA1-8B is the case that makes this concrete. It is hidden 2048 with 8 heads,
so hidden // heads says 256. ZayaConfig declares head_dim = 128, so q_proj
projects to 1024 and there are 8 heads of 128. Both answers are consistent with
a 2048-wide projector, both reshape cleanly, and only one is right - an earlier
comment in this repository took the division and was wrong about the model the
final run depends on.

The config is written out and read by the real AutoConfig rather than mocked,
because the thing under test is a getattr against an object the real loader
builds.
"""
import json

ZAYA = dict(
    model_type="qwen3_moe",
    hidden_size=2048,
    num_hidden_layers=40,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=128,
    vocab_size=262272,
    moe_intermediate_size=2048,
    num_experts=16,
    num_experts_per_tok=1,
    tie_word_embeddings=True,
)


def _model(tmp_path, **over):
    d = dict(ZAYA)
    d.update(over)
    p = tmp_path / "m"
    p.mkdir(exist_ok=True)
    (p / "config.json").write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_a_declared_head_dim_is_used_as_declared(tmp_path):
    from usaf.model_factory import detect_model
    got = detect_model(_model(tmp_path))
    assert got.head_dim == 128, got.head_dim
    # Not 2048 // 8. That number is what this parameter exists to avoid.
    assert got.head_dim != got.hidden_size // got.num_attention_heads


def test_a_config_without_head_dim_still_gets_one(tmp_path):
    from usaf.model_factory import detect_model
    cfg = _model(tmp_path)
    d = json.loads(open(cfg + r"\config.json", encoding="utf-8").read())
    del d["head_dim"]
    open(cfg + r"\config.json", "w", encoding="utf-8").write(json.dumps(d))
    got = detect_model(cfg)
    assert got.head_dim == 256, got.head_dim


def test_the_head_count_follows_the_projection_and_the_head_dim(tmp_path):
    from usaf.model_factory import detect_model
    got = detect_model(_model(tmp_path))
    q_out = got.num_attention_heads * got.head_dim
    assert q_out == 1024, q_out
    assert q_out // got.head_dim == got.num_attention_heads == 8
