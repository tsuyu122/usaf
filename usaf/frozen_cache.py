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
    """Invalidation key: sample input_ids + layer + model source.

    Each sample's length is hashed before its tokens. Without that separator,
    samples [1,2],[3] and [1],[2,3] concatenate to the same bytes and produce
    the same fingerprint, so a cache built for one segmentation would be
    accepted for the other - same N, same seq, same hidden, all checks passing,
    and the rows holding activations for entirely different samples.
    """
    h = hashlib.sha256()
    h.update(f"{detach_at}|{src}".encode())
    for s in samples:
        ids = s["input_ids"]
        arr = ids.numpy() if hasattr(ids, "numpy") else np.asarray(ids)
        flat = np.asarray(arr, dtype=np.int32).ravel()
        h.update(len(flat).to_bytes(8, "little"))
        h.update(flat.tobytes())
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

    # A build writes the array first and the metadata second, and the w+
    # open_memmap below truncates the array on the way in. A crash between the
    # two leaves the previous run's metadata describing an array that is no
    # longer there. The fingerprint still matches - same data, same layer, same
    # model - and the shape check passes, because the header was written, so the
    # cache loads with the tail of every row past the crash holding whatever was
    # in the file. Training then reads those rows as activations. Both files go
    # first, so a partial build is never mistaken for a complete one.
    for _stale in (path, path + ".json"):
        try:
            os.remove(_stale)
        except (FileNotFoundError, PermissionError, OSError):
            pass
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
