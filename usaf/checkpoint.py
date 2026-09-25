"""Sparse checkpoint save/load and model export for USAF fine-tuning.

Sparse checkpoints store only the active ~0.5% of expert weights plus
optimizer state, keeping checkpoint files tiny (MB vs GB). The export
function merges trained sparse deltas back into the original quantized
weights to produce a deployable model.
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch


def save_sparse_checkpoint(
    path: str,
    masters: dict[str, torch.nn.Parameter],
    active_idx: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    config: dict[str, Any],
    step: int,
    losses: list[float],
    train_layers: list[int],
    metric: float | None = None,
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
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != "usaf-sparse-v1":
        raise ValueError(f"Not a valid USAF sparse checkpoint: {path}")
    return state


def export_merged_weights(
    quant_path: str,
    masters: dict[str, torch.Tensor],
    active_idx: dict[str, torch.Tensor],
    output_path: str,
    group_size: int = 128,
) -> str:
    from usaf.quantization import dequantize_4bit, quantize_state_dict

    q_dict: dict[str, Any] = torch.load(quant_path, map_location="cpu", weights_only=True)

    merged_fp16: dict[str, torch.Tensor] = {}
    unsupported: list[str] = []
    for fname, entry in q_dict.items():
        if isinstance(entry, dict) and "q" in entry:
            t = dequantize_4bit(
                entry["q"], entry["s"], entry["z"], entry["shape"],
                group_size=entry.get("group_size", group_size),
            )
        elif isinstance(entry, (tuple, list)) and len(entry) == 4:
            # The loader accepts the positional (q, s, z, shape) form as
            # well. Honouring it here too, instead of skipping, is what keeps
            # an export from silently dropping tensors and producing a model
            # that looks complete but is missing experts.
            t = dequantize_4bit(
                entry[0], entry[1], entry[2], entry[3],
                group_size=group_size,
            )
        elif isinstance(entry, torch.Tensor):
            t = entry.to(torch.float16)
        else:
            unsupported.append(fname)
            continue

        if fname in masters and fname in active_idx:
            aidx = active_idx[fname].reshape(-1).to(torch.long)
            trained = masters[fname].detach().to(torch.float16).reshape(-1)
            if trained.numel() != aidx.numel():
                raise ValueError(
                    f"{fname}: active_idx has {aidx.numel()} entries but the "
                    f"trained master has {trained.numel()}"
                )
            t_flat = t.reshape(-1).clone()
            t_flat.scatter_(0, aidx, trained)
            t = t_flat.reshape(t.shape)

        merged_fp16[fname] = t

    if unsupported:
        raise ValueError(
            f"{len(unsupported)} quantized entr(y/ies) are in a format this "
            f"exporter cannot read; refusing to emit an incomplete model. "
            f"First few: {unsupported[:3]}"
        )
    if not merged_fp16:
        raise ValueError(
            f"nothing could be dequantized from {quant_path}; the export "
            f"would have been empty"
        )

    merged_q4 = quantize_state_dict(merged_fp16, group_size=group_size)
    parent = os.path.dirname(str(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(merged_q4, output_path)
    return output_path


def get_checkpoint_metadata(path: str) -> dict[str, Any]:
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
