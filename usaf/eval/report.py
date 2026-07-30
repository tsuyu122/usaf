import json
import math
from pathlib import Path
from typing import Dict
from dataclasses import asdict

from .benchmark import BenchmarkResults


def save_report(results, path, pretty=True):
    data = results.to_dict()
    indent = 2 if pretty else None
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    print(f"Report saved: {path}")
    return path


def compare_reports(before, after):
    comparison = {
        "model_before": before.model_name,
        "model_after": after.model_name,
        "datasets": {},
    }
    all_datasets = set(before.results.keys()) | set(after.results.keys())

    for ds in sorted(all_datasets):
        b_metrics = before.results.get(ds, {})
        a_metrics = after.results.get(ds, {})
        b_ppl = b_metrics.get("perplexity", float("inf"))
        a_ppl = a_metrics.get("perplexity", float("inf"))
        b_loss = b_metrics.get("loss", float("nan"))
        a_loss = a_metrics.get("loss", float("nan"))

        ppl_change = a_ppl - b_ppl if (b_ppl != float("inf") and a_ppl != float("inf")) else None
        loss_change = a_loss - b_loss if (not math.isnan(b_loss) and not math.isnan(a_loss)) else None

        comparison["datasets"][ds] = {
            "before": {"perplexity": b_ppl, "loss": b_loss},
            "after": {"perplexity": a_ppl, "loss": a_loss},
            "ppl_change": round(ppl_change, 4) if ppl_change is not None else None,
            "loss_change": round(loss_change, 6) if loss_change is not None else None,
        }

    print(f"\n{'=' * 60}")
    print(f"Comparison: {before.model_name} vs {after.model_name}")
    print(f"{'=' * 60}")
    for ds, info in comparison["datasets"].items():
        b_ppl = info["before"].get("perplexity", "?")
        a_ppl = info["after"].get("perplexity", "?")
        delta = info.get("ppl_change")
        delta_str = f"({delta:+.4f})" if delta is not None else "(N/A)"
        print(f"  {ds:25s} {b_ppl:>8.2f} -> {a_ppl:>8.2f}  {delta_str}")
    print(f"{'=' * 60}")

    return comparison
