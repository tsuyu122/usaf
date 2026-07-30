import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .perplexity import compute_perplexity
from .datasets import get_eval_texts


@dataclass
class BenchmarkConfig:
    datasets: List[str] = field(default_factory=lambda: ["synthetic-cpp"])
    dataset_paths: Dict[str, str] = field(default_factory=dict)
    max_samples: int = 64
    seq_len: int = 512
    batch_size: int = 1
    verbose: bool = True


@dataclass
class BenchmarkResults:
    config: BenchmarkConfig
    results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: float = 0.0
    elapsed: float = 0.0
    model_name: str = ""

    def print(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"Benchmark Results \u2014 {self.model_name or 'unknown'}")
        print(f"{'=' * 60}")
        for ds_name, metrics in self.results.items():
            ppl = metrics.get("perplexity", float("inf"))
            loss = metrics.get("loss", float("inf"))
            tokens = metrics.get("tokens", 0)
            tps = metrics.get("tokens_per_sec", 0)
            print(f"  {ds_name:25s} PPL {ppl:>8.2f}  loss {loss:>7.4f}  "
                  f"{tokens:>6,d} tok  {tps:>7.1f} tok/s")
        print(f"{'=' * 60}")

    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "timestamp": self.timestamp,
            "elapsed_s": self.elapsed,
            "config": asdict(self.config),
            "results": self.results,
        }


def run_benchmark(
    model,
    tokenizer,
    device,
    config=None,
    model_name="",
):
    config = config or BenchmarkConfig()
    results = {}
    t0 = time.time()

    for ds_name in config.datasets:
        ds_path = config.dataset_paths.get(ds_name)
        texts = get_eval_texts(ds_name, path=ds_path, max_samples=config.max_samples)
        if not texts:
            print(f"  [skip] {ds_name}: no texts found")
            continue

        ds_desc = ds_path.split("/")[-1] if ds_path else ds_name
        metrics = compute_perplexity(
            model, tokenizer, texts, device,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            desc=ds_desc,
            verbose=config.verbose,
        )
        results[ds_name] = metrics

        if config.verbose:
            print(f"  {ds_name}: PPL {metrics['perplexity']:.2f}  "
                  f"({metrics['tokens']:,d} tokens, {metrics['time_s']:.1f}s)")

    elapsed = time.time() - t0
    return BenchmarkResults(
        config=config,
        results=results,
        timestamp=time.time(),
        elapsed=elapsed,
        model_name=model_name,
    )
