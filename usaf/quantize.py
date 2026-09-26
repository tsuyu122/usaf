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


def collect_expert_tensors(cfg) -> dict:
    """Read the expert tensors, in whichever layout the checkpoint uses.

    Two layouts exist and they are not interchangeable.

    A checkpoint written by save_pretrained stores one tensor per expert:
    experts.0.gate_proj.weight, experts.0.up_proj.weight,
    experts.0.down_proj.weight, and so on for every expert in every layer. The
    same model, once from_pretrained has run, holds two stacked 3-D parameters
    per layer instead: gate_up_proj of shape (E, 2*inter, hidden) and down_proj
    of shape (E, hidden, inter). transformers fuses the pair on load and writes
    the pair separately on save, so the file on disk and the live model
    deliberately disagree.

    USAF wants the stacked form, so reading the stacked keys alone finds
    nothing in a checkpoint that transformers itself just wrote. The stacking
    rule is not a guess: gate_up_proj is exactly stack(cat(gate_e, up_e)),
    verified against from_pretrained at zero difference, and the reverse order
    is off by 1.3e-1 - the two projections are genuinely different tensors, so
    a zero difference here is evidence rather than coincidence.
    """
    names = [n for li in sorted(get_param_patterns(cfg))
             for n in get_param_patterns(cfg)[li]]

    shards = sorted(glob.glob(os.path.join(cfg.model_path, '*.safetensors')))
    if not shards:
        raise SystemExit(
            f'no *.safetensors in {cfg.model_path}'
            '  A from_config directory has none. Point --model at a model '
            '  that actually has weights.'
        )

    def read(keys):
        out = {}
        for shard in shards:
            from safetensors import safe_open

            with safe_open(shard, framework='pt') as sf:
                for k in keys:
                    if k not in out and k in sf.keys():
                        out[k] = sf.get_tensor(k)
            if len(out) == len(keys):
                break
        return out

    found = read(names)
    if len(found) == len(names):
        return found

    # Per-expert layout: rebuild the stacked tensors the trainer expects.
    patterns = get_param_patterns(cfg)
    n_experts = cfg.num_experts
    parts = {}
    for li, prefixes in patterns.items():
        pre = cfg.expert_prefix.format(i=li)
        gates = [f'{pre}.{e}.gate_proj.weight' for e in range(n_experts)]
        ups = [f'{pre}.{e}.up_proj.weight' for e in range(n_experts)]
        downs = [f'{pre}.{e}.down_proj.weight' for e in range(n_experts)]
        g = read(gates)
        u = read(ups)
        d = read(downs)
        if len(g) != n_experts or len(u) != n_experts:
            raise SystemExit(
                f'cannot find the expert weights of layer {li} in either the '
                f'stacked or the per-expert layout. Tried '
                f'{prefixes[0]} and {gates[0]}'
            )
        parts[f'{pre}.gate_up_proj'] = torch.stack(
            [torch.cat([g[f'{pre}.{e}.gate_proj.weight'],
                        u[f'{pre}.{e}.up_proj.weight']], dim=0)
             for e in range(n_experts)]
        )
        parts[f'{pre}.down_proj'] = torch.stack(
            [d[f'{pre}.{e}.down_proj.weight'] for e in range(n_experts)]
        )

    if not parts:
        raise SystemExit('no expert weights found in ' + cfg.model_path)
    return parts



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

    tensors = collect_expert_tensors(cfg)
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
