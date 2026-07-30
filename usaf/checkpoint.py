"""Sparse checkpoint save/load and model export for USAF fine-tuning.

Sparse checkpoints store only the active ~0.5% of expert weights plus
optimizer state, keeping checkpoint files tiny (MB vs GB). The export
function merges trained sparse deltas back into the original quantized
weights to produce a deployable model.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def save_sparse_checkpoint(
    path: str,
    masters: Dict[str, torch.nn.Parameter],
    active_idx: Dict[str, torch.Tensor],
    optimizer_state: Dict[str, Any],
    config: Dict[str, Any],
    step: int,
    losses: List[float],
    train_layers: List[int],
    metric: Optional[float] = None,
) -> str:
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "format": "usaf-sparse-v1",
        "timestamp": time.time(),
        "step": step,
        "losses": losses,
        "train_layers": train_layers,
        "config": config,
        "metric": metric,
        "active_idx": {k: v.cpu() for k, v in active_idx.items()},
        "masters": {k: v.detach().cpu() for k, v in masters.items()},
        "optimizer": optimizer_state,
    }
    torch.save(state, path)
    return path


def load_sparse_checkpoint(
    path: str,
) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != "usaf-sparse-v1":
        raise ValueError(f"Not a valid USAF sparse checkpoint: {path}")
    return state


def export_merged_weights(
    quant_path: str,
    masters: Dict[str, torch.Tensor],
    active_idx: Dict[str, torch.Tensor],
    output_path: str,
    group_size: int = 128,
) -> str:
    from usaf.quantization import dequantize_4bit, quantize_state_dict

    q_dict: Dict[str, Any] = torch.load(quant_path, map_location="cpu", weights_only=True)

    merged_fp16: Dict[str, torch.Tensor] = {}
    for fname, entry in q_dict.items():
        if isinstance(entry, dict) and "q" in entry:
            t = dequantize_4bit(
                entry["q"], entry["s"], entry["z"], entry["shape"],
                group_size=entry.get("group_size", group_size),
            )
        elif isinstance(entry, torch.Tensor):
            t = entry.to(torch.float16)
        else:
            continue

        if fname in masters and fname in active_idx:
            aidx = active_idx[fname].reshape(-1).to(torch.long)
            trained = masters[fname].detach().to(torch.float16)
            t_flat = t.reshape(-1).clone()
            t_flat.scatter_(0, aidx, trained)
            t = t_flat.reshape(t.shape)

        merged_fp16[fname] = t

    merged_q4 = quantize_state_dict(merged_fp16, group_size=group_size)
    torch.save(merged_q4, output_path)
    return output_path


def get_checkpoint_metadata(path: str) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        return {"error": "invalid checkpoint"}
    return {
        "format": state.get("format", "unknown"),
        "step": state.get("step", -1),
        "losses": state.get("losses", []),
        "n_active": sum(v.numel() for v in state.get("active_idx", {}).values()),
        "n_masters": len(state.get("masters", {})),
        "train_layers": state.get("train_layers", []),
        "timestamp": state.get("timestamp", 0),
        "config": state.get("config", {}),
        "metric": state.get("metric"),
    }
