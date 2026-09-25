"""Frozen activation cache for layers 0..DETACH_AT.

Precomputes hidden@DETACH_AT once per sample and persists to disk (fp16 memmap).
Training resumes from layer DETACH_AT+1. Decoupled from the model: build receives
a callback ``compute_hidden(sample) -> [SEQ, H]``.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable

import numpy as np
import torch


def dataset_fingerprint(samples, detach_at: int, src: str) -> str:
    """Invalidation key: sample input_ids + layer + model source."""
    h = hashlib.sha256()
    h.update(f"{detach_at}|{src}".encode())
    for s in samples:
        ids = s["input_ids"]
        arr = ids.numpy() if hasattr(ids, "numpy") else np.asarray(ids)
        h.update(np.asarray(arr, dtype=np.int32).tobytes())
    return h.hexdigest()[:16]


def load_frozen_cache(samples, seq: int, hidden: int, detach_at: int,
                      src: str, path: str) -> np.ndarray | None:
    """Return read-only memmap if fingerprint matches, else None."""
    meta_path = path + ".json"
    if not (os.path.exists(path) and os.path.exists(meta_path)):
        return None
    try:
        meta = json.load(open(meta_path))
    except Exception:
        return None
    if (meta.get("fingerprint") != dataset_fingerprint(samples, detach_at, src)
            or meta.get("seq") != seq or meta.get("hidden") != hidden
            or meta.get("N") != len(samples)):
        return None
    return np.lib.format.open_memmap(path, mode="r")


def build_frozen_cache(samples, seq: int, hidden: int, detach_at: int, src: str,
                       compute_hidden: Callable[[dict], torch.Tensor], path: str,
                       verbose: bool = True) -> np.ndarray:
    """Build (or reuse) frozen cache. ``compute_hidden(sample) -> [SEQ, H]``."""
    existing = load_frozen_cache(samples, seq, hidden, detach_at, src, path)
    if existing is not None:
        if verbose:
            print(f"  frozen cache: reusando {path} ({existing.shape})")
        return existing

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    N = len(samples)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(N, seq, hidden))
    for i, s in enumerate(samples):
        with torch.no_grad():
            h = compute_hidden(s)
        arr[i] = h.reshape(seq, hidden).detach().to("cpu", torch.float16).numpy()
        if verbose and (i % 25 == 0 or i == N - 1):
            print(f"  frozen cache build {i+1}/{N}")
    arr.flush()
    json.dump({"fingerprint": dataset_fingerprint(samples, detach_at, src),
               "N": N, "seq": seq, "hidden": hidden, "detach_at": detach_at},
              open(path + ".json", "w"))
    if verbose:
        print(f"  frozen cache: salvo {path} ({arr.shape}, {arr.nbytes/1e9:.2f}GB)")
    return arr


def get_hidden(cache: np.ndarray, idx: int, device, dtype=torch.float16) -> torch.Tensor:
    """Return hidden@DETACH_AT as [1, SEQ, H] on the target device.

    Raises IndexError when ``idx`` is outside the cache. Callers relied on that
    to fall back to a full forward pass, but the fallback was silent: a sample
    whose index was not in the cache simply stopped being evaluated, and an
    evaluation over no samples at all returned inf rather than failing.
    """
    n = len(cache)
    if not isinstance(idx, (int,)) or isinstance(idx, bool):
        idx = int(idx)
    if idx < 0:
        idx += n
    if idx < 0 or idx >= n:
        raise IndexError(
            f"frozen cache index {idx} is out of range for a cache holding "
            f"{n} samples; this sample was never cached"
        )
    arr = np.array(cache[idx], copy=True)
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)
    return t.unsqueeze(0)
