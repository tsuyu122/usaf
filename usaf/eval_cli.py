"""USAF Evaluation CLI — benchmark any model without training.

Usage:
    python -m usaf.eval_cli --model Qwen/Qwen3-30B-A3B
    python -m usaf.eval_cli --model Qwen3-30B-A3B-q4 --datasets synthetic-cpp,wikitext-2
    python -m usaf.eval_cli --model my-model --report eval_results.json
"""
import argparse
import json
from pathlib import Path

import torch

from .eval.benchmark import BenchmarkConfig, BenchmarkResults, run_benchmark
from .eval.report import save_report


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
        # The checkpoint keys are the full parameter names, so they match the
        # model keys directly. The previous code compared with
        # n.endswith('.' + fname), which prepends a dot to an already-fully
        # qualified name: no key could ever match, the loop fell through, and
        # the base model was evaluated and reported as if it were fine-tuned.
        params = dict(model.named_parameters())
        applied = 0
        missing: list[str] = []
        for fname, aidx in ckpt.get("active_idx", {}).items():
            trained = ckpt.get("masters", {}).get(fname)
            if trained is None:
                missing.append(f"{fname} (no master)")
                continue
            p = params.get(fname)
            if p is None:
                missing.append(f"{fname} (not in model)")
                continue
            aidx = aidx.reshape(-1).to(torch.long)
            if aidx.numel() == 0:
                missing.append(f"{fname} (no active elements)")
                continue
            # scatter_ requires self.dtype == src.dtype, and the checkpoint
            # masters are stored in the dtype they were saved with, which is
            # not necessarily the dtype the model was loaded in.
            trained = trained.reshape(-1).to(p.dtype)
            if trained.numel() != aidx.numel():
                raise ValueError(
                    f"{fname}: checkpoint has {aidx.numel()} active indices "
                    f"but {trained.numel()} trained values"
                )
            if int(aidx.max()) >= p.numel():
                raise ValueError(
                    f"{fname}: active index {int(aidx.max())} is outside the "
                    f"parameter ({p.numel()} elements) - the checkpoint does "
                    f"not belong to this model"
                )
            with torch.no_grad():
                # reshape(-1) is only a view when the parameter is contiguous.
                # On a non-contiguous one it silently copies, and the
                # in-place scatter would be written to that throwaway copy
                # and lost. The previous copy_(reshape(-1).scatter(...))
                # instead raised a shape mismatch on every expert tensor.
                flat = p.data.reshape(-1)
                if flat.data_ptr() != p.data.data_ptr():
                    updated = flat.scatter(0, aidx, trained).reshape(p.shape)
                    p.data = updated
                else:
                    flat.scatter_(0, aidx, trained)
            applied += 1
        if missing:
            raise KeyError(
                f"{len(missing)} checkpoint tensor(s) do not match the model; "
                f"refusing to evaluate a partially applied checkpoint. "
                f"First few: {missing[:3]}"
            )
        if applied == 0:
            raise ValueError(
                "the checkpoint contained no active tensors that matched the "
                "model; nothing was applied"
            )
        print(f"  Checkpoint applied to {applied} tensors "
              f"(step {ckpt.get('step', '?')})")
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
            with open(prev_path) as f:
                prev_data = json.load(f)
            prev_results = BenchmarkResults(
                config=BenchmarkConfig(**prev_data.get("config", {})),
                results=prev_data.get("results", {}),
                model_name=prev_data.get("model", "previous"),
            )
            compare_reports(prev_results, results)


if __name__ == "__main__":
    main()
