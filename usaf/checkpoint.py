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
    routers: dict[str, Any] | None = None,
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
        "routers": {k: v.detach().cpu()
                    for k, v in (routers or {}).items()},
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


def apply_trained_entries(
    t: torch.Tensor,
    aidx: torch.Tensor,
    trained: torch.Tensor,
    name: str = "",
) -> torch.Tensor:
    """Write the trained entries back into a dense tensor, and return it.

    ``active_idx`` holds flat positions into the WHOLE expert tensor - all 32
    of them, not one - so the tensor has to be flattened before it can be
    indexed by them and reshaped back afterwards. The measured range settles
    it: for a [32, 1024, 1024] gate_up the largest index is 33554426 against
    a flattened size of 33554432, which is the last slot of the last expert,
    and no index into one expert reaches that.

    Two callers put trained weights back into a dense tensor and they
    disagreed about this. The merged export flattened; the huggingface export
    indexed the unflattened tensor and died with an index of 24260437 out of
    bounds for a dimension of 32 - the expert count, because the index was
    global and the tensor was not - after five hours of training that was
    entirely fine. One function, called by both.
    """
    aidx = aidx.reshape(-1).to(torch.long)
    trained = trained.detach().reshape(-1)
    if trained.numel() != aidx.numel():
        raise ValueError(
            f"{name}: active_idx has {aidx.numel()} entries but the "
            f"trained master has {trained.numel()}"
        )
    if aidx.numel() and int(aidx.max()) >= t.numel():
        raise ValueError(
            f"{name}: active index {int(aidx.max())} is outside the "
            f"tensor ({t.numel()} elements)"
        )
    t_flat = t.reshape(-1).clone()
    # scatter_ requires matching dtypes; the dequantized tensor and the stored
    # master do not necessarily share one.
    t_flat.scatter_(0, aidx, trained.to(t_flat.dtype))
    return t_flat.reshape(t.shape)


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
    # fname -> boolean mask over quantization groups, True where a trained
    # weight lives. Collected here and used after requantization.
    touched_groups: dict[str, torch.Tensor] = {}
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

        aidx = active_idx.get(fname)
        if fname in masters and aidx is not None:
            t = apply_trained_entries(t, aidx, masters[fname], fname)
            # The mask below is built per trained tensor, and it needs the
            # flat form; keeping it here is why the loop only ever marked
            # the groups a run actually wrote to.
            t_flat = t.reshape(-1)

            _gs = group_size
            if isinstance(entry, dict) and "group_size" in entry:
                _gs = int(entry["group_size"])
            elif isinstance(entry, (tuple, list)) and len(entry) == 4:
                _gs = group_size
            _n_groups = (t_flat.numel() + _gs - 1) // _gs
            _m = torch.zeros(t_flat.numel(), dtype=torch.bool)
            _m[aidx] = True
            _pad = _n_groups * _gs - t_flat.numel()
            if _pad:
                _m = torch.cat([_m, torch.zeros(_pad, dtype=torch.bool)])
            touched_groups[fname] = _m.view(_n_groups, _gs).any(dim=1)

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

    # Requantizing a whole tensor recomputes every group's scale and zero, so
    # groups the training never touched came back with slightly different
    # dequantized values - the dequantize/requantize round trip is not exact.
    # On the fixture that moved 8464 of 128451 untouched weights, by up to
    # 4.3e-04, which contradicts the promise that an export leaves everything
    # outside the active set bit-identical.
    #
    # A group that contains no trained weight cannot have changed, so its
    # original q, s and z are restored verbatim. Only groups that actually
    # hold a trained value are re-derived, because those have to be, or the
    # training would be clamped away by the old scale.
    merged_q4 = _restore_untouched_groups(
        merged_q4, q_dict, merged_fp16, touched_groups, group_size
    )

    parent = os.path.dirname(str(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(merged_q4, output_path)
    return output_path


def _restore_untouched_groups(
    merged_q4: dict[str, Any],
    original: dict[str, Any],
    _unused: dict[str, torch.Tensor],
    touched: dict[str, torch.Tensor],
    default_group_size: int,
) -> dict[str, Any]:
    """Put back the original q/s/z for every group the training never touched.

    The dequantize-then-requantize round trip is not exact, so re-deriving a
    group nobody trained on still moves its weights. Restoring the original
    bytes for those groups is both cheaper and more faithful.
    """
    for fname, mask in touched.items():
        new_entry = merged_q4.get(fname)
        old_entry = original.get(fname)

        # The freshly quantized side is always a dict, but the file on disk
        # may hold either shape: {"q","s","z"} or the positional (q, s, z,
        # shape) tuple. Normalise the old side, or a checkpoint written in the
        # tuple form silently skips the whole restoration.
        if isinstance(old_entry, (tuple, list)) and len(old_entry) == 4:
            old_entry = {
                "q": old_entry[0], "s": old_entry[1],
                "z": old_entry[2], "shape": old_entry[3],
            }
        if not isinstance(new_entry, dict) or not isinstance(old_entry, dict):
            continue
        if not ({"q", "s", "z"} <= set(new_entry) and {"q", "s", "z"} <= set(old_entry)):
            continue
        if mask.numel() != new_entry["s"].numel():
            # Different group layout; leave it alone rather than guess.
            continue

        keep = (~mask.to(torch.bool))
        if not bool(keep.any()):
            continue

        for k in ("s", "z"):
            merged_k = new_entry[k].reshape(-1).clone()
            merged_k[keep] = old_entry[k].reshape(-1)[keep].to(merged_k.dtype)
            new_entry[k] = merged_k.reshape(new_entry[k].shape)

        # q packs two values per byte, so a whole group is recoverable only if
        # the group holds an even number of values. Every group_size in use is
        # even, and when it is not we leave q alone rather than write bytes
        # that straddle two groups.
        gs = int(new_entry.get("group_size") or default_group_size)
        if gs % 2 == 0:
            bytes_per_group = gs // 2
            nq = new_entry["q"].numel()
            if nq == int(keep.numel()) * bytes_per_group:
                n_groups = int(keep.numel())
                q_new = new_entry["q"].reshape(n_groups, bytes_per_group).clone()
                q_old = old_entry["q"].reshape(n_groups, bytes_per_group)
                q_new[keep] = q_old[keep].to(q_new.dtype)
                new_entry["q"] = q_new.reshape(new_entry["q"].shape)

    return merged_q4


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
