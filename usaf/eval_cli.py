"""USAF Evaluation CLI — benchmark any model without training.

Usage:
    python -m usaf.eval_cli --model Qwen/Qwen3-30B-A3B
    python -m usaf.eval_cli --model Qwen3-30B-A3B-q4 --datasets synthetic-cpp,wikitext-2
    python -m usaf.eval_cli --model my-model --report eval_results.json
"""
import argparse
import json
import time
import torch
from pathlib import Path

from .eval.benchmark import run_benchmark, BenchmarkConfig
from .eval.report import save_report
from .eval.datasets import SYNTHETIC_TEXTS


def build_parser():
    p = argparse.ArgumentParser(description="USAF Evaluation CLI")
    p.add_argument("--model", type=str, default="", help="Model ID or path")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Sparse checkpoint to evaluate (requires --model for base weights)")
    p.add_argument("--datasets", type=str, default="synthetic-cpp",
                   help="Comma-separated dataset names")
    p.add_argument("--max-samples", type=int, default=64,
                   help="Max samples per dataset")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu",
                   help="Device: cpu, cuda, or dml")
    p.add_argument("--report", type=str, default="",
                   help="Path to save JSON report")
    p.add_argument("--compare", type=str, default="",
                   help="Previous report JSON to compare against")
    return p


def main():
    ns = build_parser().parse_args()

    device = torch.device(ns.device)
    model = None
    tokenizer = None

    if ns.model:
        print(f"Loading model: {ns.model}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ns.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            ns.model, torch_dtype=torch.float16, trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        print(f"  Model loaded: {model.__class__.__name__}")

    if ns.checkpoint and ns.model:
        print(f"Loading checkpoint: {ns.checkpoint}")
        ckpt = torch.load(ns.checkpoint, map_location="cpu", weights_only=True)
        for fname, aidx in ckpt.get("active_idx", {}).items():
            fname_full = fname
            trained = ckpt["masters"][fname]
            for n, p in model.named_parameters():
                if n.endswith("." + fname):
                    p.data = p.data.reshape(-1)
                    p.data.scatter_(0, aidx.to(torch.long), trained.to(p.dtype))
                    p.data = p.data.reshape(
                        ckpt.get("shapes", {}).get(fname, p.data.shape)
                    )
                    break
        print(f"  Checkpoint applied (step {ckpt.get('step', '?')})")
    elif ns.checkpoint:
        print("ERROR: --model is required with --checkpoint")
        return

    ds_list = [d.strip() for d in ns.datasets.split(",") if d.strip()]
    cfg = BenchmarkConfig(
        datasets=ds_list,
        max_samples=ns.max_samples,
        seq_len=ns.seq_len,
        batch_size=ns.batch_size,
    )

    results = run_benchmark(model, tokenizer, device, cfg, model_name=ns.model or ns.checkpoint)
    results.print()

    if ns.report:
        path = save_report(results, ns.report)
        print(f"Report saved: {path}")

    if ns.compare:
        from .eval.report import compare_reports
        prev_path = Path(ns.compare)
        if prev_path.exists():
            with open(prev_path, "r") as f:
                prev_data = json.load(f)
            prev_results = BenchmarkResults(
                config=BenchmarkConfig(**prev_data.get("config", {})),
                results=prev_data.get("results", {}),
                model_name=prev_data.get("model", "previous"),
            )
            compare_reports(prev_results, results)


if __name__ == "__main__":
    main()
