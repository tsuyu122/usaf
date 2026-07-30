"""Tests for the eval package."""

import math
import os
import pytest
import torch
from unittest.mock import MagicMock


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.eval.return_value = None
    return model


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.randint(2, 100, (1, 16)),
    }
    return tok


def test_get_eval_texts_synthetic():
    from usaf.eval.datasets import get_eval_texts, SYNTHETIC_TEXTS
    import os
    texts = get_eval_texts("synthetic-cpp", max_samples=4)
    assert len(texts) == 4
    assert isinstance(texts[0], str)


def test_get_eval_texts_default():
    from usaf.eval.datasets import get_eval_texts
    texts = get_eval_texts(max_samples=2)
    assert len(texts) == 2


def test_synthetic_cpp_dataset():
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.randint(2, 100, (1, 32))}
    from usaf.eval.datasets import SyntheticCppDataset
    ds = SyntheticCppDataset(tok, seq_len=32, max_samples=4)
    assert len(ds) == 4
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item


def test_eval_dataset():
    from usaf.eval.datasets import EvalDataset
    texts = ["hello world", "test text"]
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.randint(2, 100, (1, 16))}
    ds = EvalDataset(texts, tok, name="test")
    assert ds.name == "test"
    assert len(ds) == 2


@pytest.mark.slow
def test_compute_perplexity_synthetic_model():
    """Integration test with a tiny random transformer."""
    from transformers import AutoConfig
    from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
    from usaf.eval.perplexity import compute_perplexity

    cfg = AutoConfig.for_model("gpt2", vocab_size=256, n_embd=64, n_layer=2, n_head=4)
    model = GPT2LMHeadModel(cfg)
    model.eval()

    class TinyTok:
        pad_token = None
        eos_token = "[EOS]"
        def __call__(self, text, truncation=False, max_length=512, return_tensors=None):
            ids = [ord(c) % 254 + 2 for c in text[:max_length]]
            while len(ids) < 4:
                ids.append(2)
            return {"input_ids": torch.tensor([ids[:max_length]], dtype=torch.long)}

    device = torch.device("cpu")
    texts = ["hello world", "test evaluation text"]

    result = compute_perplexity(model, TinyTok(), texts, device, seq_len=32, verbose=False)
    assert "loss" in result
    assert "perplexity" in result
    assert result["perplexity"] > 0
    assert not math.isnan(result["loss"])
    assert result["samples"] == 2


def test_benchmark_config_defaults():
    from usaf.eval.benchmark import BenchmarkConfig
    cfg = BenchmarkConfig()
    assert cfg.datasets == ["synthetic-cpp"]
    assert cfg.max_samples == 64
    assert cfg.verbose is True


def test_benchmark_results():
    from usaf.eval.benchmark import BenchmarkResults, BenchmarkConfig
    cfg = BenchmarkConfig(max_samples=4, verbose=False)
    results = BenchmarkResults(
        config=cfg,
        results={"synthetic-cpp": {"perplexity": 10.5, "loss": 2.35, "tokens": 100, "tokens_per_sec": 50.0}},
        model_name="test-model",
    )
    d = results.to_dict()
    assert d["model"] == "test-model"
    assert d["results"]["synthetic-cpp"]["perplexity"] == 10.5


def test_save_and_compare_reports(tmp_path):
    from usaf.eval.benchmark import BenchmarkResults, BenchmarkConfig
    from usaf.eval.report import save_report, compare_reports

    cfg = BenchmarkConfig(max_samples=4, verbose=False)
    before = BenchmarkResults(
        config=cfg,
        results={"synth": {"perplexity": 12.0, "loss": 2.5, "tokens": 50}},
        model_name="model-before",
    )
    after = BenchmarkResults(
        config=cfg,
        results={"synth": {"perplexity": 8.0, "loss": 2.1, "tokens": 50}},
        model_name="model-after",
    )

    # Save report
    path = str(tmp_path / "report.json")
    save_report(before, path)
    assert os.path.exists(path)

    # Compare
    comp = compare_reports(before, after)
    assert "datasets" in comp
    assert comp["datasets"]["synth"]["ppl_change"] == -4.0
    assert comp["datasets"]["synth"]["loss_change"] == -0.4


def test_compare_reports_empty_datasets():
    from usaf.eval.benchmark import BenchmarkResults, BenchmarkConfig
    from usaf.eval.report import compare_reports

    cfg = BenchmarkConfig(max_samples=4, verbose=False)
    r1 = BenchmarkResults(config=cfg, results={}, model_name="empty")
    r2 = BenchmarkResults(config=cfg, results={}, model_name="also-empty")
    comp = compare_reports(r1, r2)
    assert comp["datasets"] == {}


def test_run_benchmark_with_mock_model():
    from usaf.eval.benchmark import run_benchmark, BenchmarkConfig

    model = MagicMock()
    model.eval.return_value = None
    tok = MagicMock()

    def tok_side_effect(text, **kwargs):
        ids = [ord(c) % 254 + 2 for c in text[:kwargs.get("max_length", 512)]]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    tok.side_effect = tok_side_effect
    model.return_value = MagicMock(loss=torch.tensor(2.5))

    device = torch.device("cpu")
    cfg = BenchmarkConfig(max_samples=4, seq_len=32, verbose=False)
    results = run_benchmark(model, tok, device, cfg, model_name="mock")
    assert results.model_name == "mock"
    assert "synthetic-cpp" in results.results
