"""Quantize the expert weights of a model to 4-bit: the input the trainer needs.

USAF trains expert tensors in 4-bit and the training CLI refuses to start
without an experts_q4.pt. That file had no producer anywhere in the project:
the root trainer loads one that already exists, and the library exposes
quantize_state_dict but nothing that gathers a model and calls it. So the
universal CLI could not be pointed at a fresh model from HuggingFace - the
documented workflow stopped one step short, at exactly the point where you
would notice on a new machine.

This is that step. It reads the safetensors shards, keeps only the expert
tensors the trainer will ask for, quantizes them per group, and writes
<out>/experts_q4.pt, which is what --quant-path auto-detects.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

from usaf.model_factory import detect_model, get_param_patterns
from usaf.quantization import quantize_state_dict


def collect_expert_tensors(model_path: str, names: list[str]) -> dict:
    """Read exactly the named tensors out of the safetensors shards.

    Only those are loaded. A 30B checkpoint is mostly experts, but the
    embeddings and attention are big enough that reading everything would be
    a multi-gigabyte allocation for tensors that are thrown away immediately.
    """
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise SystemExit(
            f'no *.safetensors in {model_path}'
            '  A from_config directory has none. Point --model at a model '
            '  that actually has weights.'
        )

    want = set(names)
    found = {}
    for shard in shards:
        from safetensors import safe_open

        with safe_open(shard, framework='pt') as sf:
            for key in sf.keys():
                if key in want and key not in found:
                    found[key] = sf.get_tensor(key)
        if len(found) == len(want):
            break

    missing = sorted(want - set(found))
    if missing:
        raise SystemExit(
            f'the checkpoint is missing {len(missing)} of {len(want)} '
            f'expert tensors. First few: {missing[:3]}'
            '  The experts are not named the way the config reports them.'
        )
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m usaf.quantize",
        description="Quantize the expert weights of a model to 4-bit for USAF.",
    )
    p.add_argument("--model", required=True, help="model directory or id")
    p.add_argument("--out", default="", help="output directory")
    p.add_argument('--group-size', type=int, default=128)
    p.add_argument('--layers', default='', help='comma-separated layer indices')
    args = p.parse_args(argv)

    model_path = args.model
    if os.path.isdir(model_path):
        if not os.path.exists(os.path.join(model_path, 'config.json')):
            raise SystemExit(f'no config.json in {model_path}')
    else:
        from huggingface_hub import snapshot_download

        print(f'downloading {model_path}', flush=True)
        model_path = snapshot_download(
            model_path, allow_patterns=['*.json', '*.safetensors', 'tokenizer*']
        )

    cfg = detect_model(model_path)
    if not cfg.is_moe:
        raise SystemExit(
            'not a mixture-of-experts model, so there are no expert '
            'weights to quantize'
        )

    patterns = get_param_patterns(cfg)
    if args.layers:
        keep = {int(x) for x in args.layers.split(',') if x.strip()}
        patterns = {k: v for k, v in patterns.items() if k in keep}

    names = [n for li in sorted(patterns) for n in patterns[li]]
    print(
        f'model: {cfg.num_layers} layers, {cfg.num_experts} experts, '
        f'top-{cfg.num_experts_per_tok}, intermediate '
        f'{cfg.expert_intermediate}'
    )
    print(
        f'quantizing {len(names)} expert tensors '
        f'(group_size {args.group_size})'
    )

    tensors = collect_expert_tensors(model_path, names)
    params = sum(t.numel() for t in tensors.values())
    print(f'  {params:,} elements, {params * 2 / 1e9:.2f} GB fp16', flush=True)

    q_dict = quantize_state_dict(tensors, group_size=args.group_size)

    out_dir = args.out or (model_path.rstrip('/\\') + '-q4')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'experts_q4.pt')
    torch.save(q_dict, out_file)

    packed = os.path.getsize(out_file)
    ratio = (params * 2) / max(packed, 1)
    print(f'wrote {out_file} ({packed / 1e9:.2f} GB, {ratio:.2f}x smaller)',
          flush=True)
    print('train it with:', flush=True)
    print(f'  python -m usaf.train --model {model_path} --dataset <data.jsonl>',
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
