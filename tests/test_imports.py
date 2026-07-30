"""Verify all core usaf package imports work correctly."""


def test_package_import():
    import usaf
    assert hasattr(usaf, "USAFConfig")
    assert hasattr(usaf, "CppDataset")
    assert hasattr(usaf, "ImportanceScorer")
    assert hasattr(usaf, "TopKSelector")
    assert hasattr(usaf, "SparseAdam")
    assert hasattr(usaf, "USAFFineTuner")
    assert hasattr(usaf, "Evaluator")


def test_config_import():
    from usaf.config import USAFConfig
    cfg = USAFConfig()
    assert cfg.model_id == "local_model_qwen"
    assert cfg.initial_active_k == 400_000
    assert cfg.reselect_every_n_steps == 500
    assert cfg.learning_rate == 1e-4
    assert cfg.seed == 42


def test_sparse_optim_import():
    from usaf.sparse_optim import SparseAdam
    import torch


def test_selector_import():
    from usaf.selector import TopKSelector, ThresholdSelector, DynamicSelector


def test_importance_import():
    from usaf.importance import ImportanceScorer


def test_utils_import():
    from usaf.utils import count_parameters, estimate_optimizer_memory
    import torch
    model = torch.nn.Linear(10, 10)
    n = count_parameters(model)
    assert n == 110
    mem = estimate_optimizer_memory(100)
    assert mem > 0


def test_model_factory_import():
    from usaf.model_factory import MoEConfig, detect_model
