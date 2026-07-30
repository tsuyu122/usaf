"""Tests for USAFConfig dataclass."""
from pathlib import Path

from usaf.config import USAFConfig


def test_default_config():
    cfg = USAFConfig()
    assert cfg.model_id == "local_model_qwen"
    assert cfg.context_length == 2048
    assert cfg.initial_active_k == 400_000
    assert cfg.reselect_every_n_steps == 500
    assert cfg.learning_rate == 1e-4
    assert cfg.seed == 42


def test_config_custom_values():
    cfg = USAFConfig(
        model_id="test-model",
        initial_active_k=100_000,
        learning_rate=5e-4,
        batch_size=4,
    )
    assert cfg.model_id == "test-model"
    assert cfg.initial_active_k == 100_000
    assert cfg.learning_rate == 5e-4
    assert cfg.batch_size == 4


def test_resolve_paths():
    cfg = USAFConfig(
        checkpoint_dir="custom_ckpt",
        log_dir="custom_logs",
    )
    base = Path("/tmp/test_base")
    resolved = cfg.resolve_paths(base)
    assert str(resolved.checkpoint_dir) == str(base / "custom_ckpt")
    assert str(resolved.log_dir) == str(base / "custom_logs")
    assert str(resolved.cache_dir) == str(base / "data/activation_cache")


def test_resolve_paths_default_base():
    cfg = USAFConfig()
    resolved = cfg.resolve_paths()
    cwd = Path.cwd()
    assert str(resolved.checkpoint_dir) == str(cwd / "checkpoints")
    assert str(resolved.log_dir) == str(cwd / "logs")


def test_config_cpp_extensions():
    cfg = USAFConfig()
    assert ".cpp" in cfg.cpp_extensions
    assert ".h" in cfg.cpp_extensions
    assert ".py" not in cfg.cpp_extensions


def test_config_betas():
    cfg = USAFConfig()
    assert cfg.betas == (0.9, 0.999)
    assert cfg.eps == 1e-8
