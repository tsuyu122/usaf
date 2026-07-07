"""HQQ-style 4-bit quantization for expert weights (Qwen3-30B-A3B).

Pure PyTorch — no CUDA, no bitsandbytes, no external extensions.
Per-group quantization with fp16 scale and zero point.
Two 4-bit values packed per int8: low nibble = first, high nibble = second.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator gradient."""
    return (x - x.detach()) + x.round()


def quantize_4bit(
    tensor: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    """Quantize a float tensor to 4-bit per-group HQQ format.

    Each group of ``group_size`` elements gets its own fp16 scale and zero point.
    Values are clamped to [0, 15] then packed two-per-byte into an int8 tensor.

    Args:
        tensor: Float tensor to quantize (will be made contiguous if needed).
        group_size: Number of elements per quantization group.

    Returns:
        q_int4:  Packed int8 tensor (int8, shape = (groups, ceil(group_size/2))).
        scale:   Float16 scale per group (shape = (groups,)).
        zero:    Float16 zero per group (shape = (groups,)).
        shape:   Original tensor shape for dequantization.
    """
    if tensor.dim() == 0:
        raise ValueError("Scalar tensors cannot be quantized; skip them explicitly.")

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    original_shape = tensor.shape
    x = tensor.flatten().float()
    numel = x.numel()

    effective_gs = min(group_size, numel)
    num_groups = (numel + effective_gs - 1) // effective_gs
    padded = num_groups * effective_gs

    if padded > numel:
        x = torch.cat([x, torch.zeros(padded - numel, device=x.device, dtype=x.dtype)])

    x_2d = x.view(num_groups, effective_gs)

    x_min = x_2d.amin(dim=1, keepdim=True)
    x_max = x_2d.amax(dim=1, keepdim=True)

    scale = (x_max - x_min) / 15.0
    scale = scale.clamp_min(1e-12)
    zero = x_min

    q_float = _round_ste((x_2d - zero) / scale)
    q_float = q_float.clamp(0, 15)
    q_int = q_float.to(torch.uint8)

    packed = torch.empty(
        (num_groups, (effective_gs + 1) // 2), dtype=torch.int8, device="cpu"
    )

    even_vals = q_int[:, 0::2]
    odd_vals = torch.zeros_like(even_vals)
    if effective_gs > 1:
        odd_vals[:, : q_int[:, 1::2].shape[1]] = q_int[:, 1::2]

    packed_raw = even_vals | (odd_vals << 4)
    packed.copy_(packed_raw.view(torch.int8))

    scale_fp16 = scale.squeeze(1).to(torch.float16)
    zero_fp16 = zero.squeeze(1).to(torch.float16)

    return packed, scale_fp16, zero_fp16, original_shape


def dequantize_4bit(
    q_int4: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    shape: torch.Size,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize a packed 4-bit tensor back to float16.

    Args:
        q_int4: Packed int8 tensor (shape = (groups, ceil(group_size/2))).
        scale:  Float16 scale per group (shape = (groups,)).
        zero:   Float16 zero per group (shape = (groups,)).
        shape:  Original tensor shape.
        group_size: Number of elements per quantization group.

    Returns:
        Dequantized float16 tensor with original shape.
    """
    num_groups = q_int4.shape[0]
    packed_half = q_int4.shape[1]

    raw = q_int4.to(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    del raw

    interleaved = torch.empty((num_groups, packed_half * 2), dtype=torch.uint8)
    interleaved[:, 0::2] = low
    interleaved[:, 1::2] = high
    del low, high

    original_numel = 1
    for s in shape:
        original_numel *= s

    effective_gs = min(group_size, original_numel)
    actual_numel = num_groups * effective_gs

    q = interleaved.flatten()[:actual_numel].view(num_groups, effective_gs).to(torch.float16)
    del interleaved
    x = q * scale.to(torch.float16).unsqueeze(1) + zero.to(torch.float16).unsqueeze(1)
    del q

    return x.flatten()[:original_numel].view(shape)


def quantize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    group_size: int = 128,
    min_size: int = 256,
) -> Dict[str, Any]:
    """Quantize all tensors in a state dict above ``min_size`` elements.

    Scalar tensors (0-dim) and small tensors are left as-is (float16).

    Each quantized entry is stored as a dict with keys "q", "s", "z", "shape".
    Other parameters are kept as plain torch.Tensor.

    Args:
        state_dict: Model state dict (tensor name → tensor).
        group_size: Group size for per-group quantization.
        min_size:   Minimum number of elements to quantize.

    Returns:
        Dict with the same keys; quantized entries are sub-dicts.
    """
    out: Dict[str, Any] = {}

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            out[name] = param
            continue

        if param.dim() == 0 or param.numel() < min_size:
            out[name] = param.detach().to(torch.float16)
            continue

        q, s, z, shape = quantize_4bit(param, group_size=group_size)
        out[name] = {"q": q, "s": s, "z": z, "shape": shape, "group_size": group_size}

    return out


def dequantize_state_dict(q_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Dequantize a state dict produced by ``quantize_state_dict``.

    Args:
        q_dict: Quantized state dict (same format as output of quantize_state_dict).

    Returns:
        Plain state dict with all tensors as float16.
    """
    out: Dict[str, torch.Tensor] = {}

    for name, entry in q_dict.items():
        if isinstance(entry, dict) and "q" in entry:
            group_size = entry.get("group_size", 128)
            t = dequantize_4bit(
                entry["q"], entry["s"], entry["z"], entry["shape"],
                group_size=group_size,
            )
            out[name] = t
        elif isinstance(entry, torch.Tensor):
            out[name] = entry.to(torch.float16)
        else:
            out[name] = entry

    return out


def estimate_quantized_size(num_params: int, bits: int = 4) -> Dict[str, float]:
    """Estimate memory footprint for a quantized tensor.

    Args:
        num_params: Number of parameters (elements).
        bits:       Bit width of quantized values (default 4).

    Returns:
        Dict with keys "weight_mb", "scale_mb", "zero_mb", "total_mb" in MiB.
    """
    to_mib = 1.0 / (1024 * 1024)
    weight_bytes = math.ceil(num_params * bits / 8)
    weight_mb = weight_bytes * to_mib

    scale_bytes = num_params * 2
    scale_mb = scale_bytes * to_mib

    zero_mb = scale_mb
    total_mb = weight_mb + scale_mb + zero_mb

    return {
        "weight_mb": round(weight_mb, 4),
        "scale_mb": round(scale_mb, 4),
        "zero_mb": round(zero_mb, 4),
        "total_mb": round(total_mb, 4),
    }


def estimate_quantized_state_dict_size(
    state_dict: Dict[str, torch.Tensor],
    group_size: int = 128,
    min_size: int = 256,
) -> Dict[str, Any]:
    """Estimate memory savings from quantizing a state dict.

    Args:
        state_dict: Model state dict.
        group_size: Group size for per-group quantization.
        min_size:   Only tensors >= this many elements are considered for quantization.

    Returns:
        Dict with original_mb, quantized_mb, compression_ratio, per_param breakdown.
    """
    to_mib = 1.0 / (1024 * 1024)
    original_bytes = 0
    quantized_bytes = 0
    per_param: list[dict] = []

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor) or param.dim() == 0:
            continue

        n = param.numel()
        orig_b = n * param.element_size()

        if n < min_size:
            original_bytes += orig_b
            quantized_bytes += orig_b
            continue

        gs = min(group_size, n)
        groups = math.ceil(n / gs)

        q_b = groups * ((gs + 1) // 2)
        s_b = groups * 2
        z_b = s_b
        q_total = q_b + s_b + z_b

        original_bytes += orig_b
        quantized_bytes += q_total
        per_param.append({
            "name": name,
            "elements": n,
            "orig_mb": round(orig_b * to_mib, 4),
            "quant_mb": round(q_total * to_mib, 4),
        })

    original_mb = original_bytes * to_mib
    quantized_mb = quantized_bytes * to_mib
    ratio = original_mb / quantized_mb if quantized_mb > 0 else 1.0

    return {
        "original_mb": round(original_mb, 2),
        "quantized_mb": round(quantized_mb, 2),
        "compression_ratio": round(ratio, 2),
        "per_param": per_param,
    }


def reconstruction_error(
    original: torch.Tensor,
    dequantized: torch.Tensor,
) -> float:
    """Compute relative mean squared error between original and dequantized tensors.

    Args:
        original:    Original float tensor (before quantization).
        dequantized: Dequantized tensor (after round-trip).

    Returns:
        Relative MSE: ``mse(dequant, orig) / variance(orig)``.
        Returns 0.0 if original has zero variance.
    """
    orig = original.detach().float()
    deq = dequantized.detach().float()

    if deq.shape != orig.shape:
        deq = deq.view(orig.shape)

    sq_err = (deq - orig) ** 2
    mse = sq_err.mean().item()

    var = orig.var(unbiased=False).item()
    if var < 1e-12:
        return 0.0

    return mse / var


def _compute_per_group_minmax(
    x_2d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group min and max."""
    x_min = x_2d.amin(dim=1, keepdim=True)
    x_max = x_2d.amax(dim=1, keepdim=True)
    return x_min, x_max


def quantize_with_outliers(
    tensor: torch.Tensor,
    group_size: int = 128,
    outlier_fraction: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """HQQ-style quantization with outlier sparsification.

    Outliers are stored separately in fp16. Only the remaining values
    are quantized to 4-bit, which reduces error for heavy-tailed distributions.

    Args:
        tensor:           Float tensor to quantize.
        group_size:       Group size for quantization.
        outlier_fraction: Fraction of elements treated as outliers (top-k by magnitude).

    Returns:
        Dict with:
          - "q": packed int8 weights
          - "s": fp16 scale per group
          - "z": fp16 zero per group
          - "shape": original shape
          - "outlier_indices": int64 flat indices of outliers
          - "outlier_values": fp16 outlier values
    """
    if tensor.dim() == 0:
        raise ValueError("Scalar tensors cannot be quantized.")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    shape = tensor.shape
    x = tensor.flatten().float()
    numel = x.numel()

    num_outliers = max(1, int(numel * outlier_fraction))
    _, top_indices = torch.topk(x.abs(), num_outliers)
    outlier_vals = x[top_indices].to(torch.float16)

    mask = torch.ones(numel, dtype=torch.bool, device=x.device)
    mask[top_indices] = False
    x_clean = x[mask]

    q, s, z, _ = quantize_4bit(x_clean.view(-1), group_size=group_size)

    return {
        "q": q,
        "s": s,
        "z": z,
        "shape": shape,
        "group_size": group_size,
        "outlier_indices": top_indices.to(torch.int64),
        "outlier_values": outlier_vals,
    }


def dequantize_with_outliers(packed: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Dequantize a tensor packed by ``quantize_with_outliers``."""
    q = packed["q"]
    s = packed["s"]
    z = packed["z"]
    shape = packed["shape"]
    group_size = packed.get("group_size", 128)
    outlier_idx = packed["outlier_indices"]
    outlier_val = packed["outlier_values"]

    numel = 1
    for dim in shape:
        numel *= dim

    n_outliers = outlier_idx.numel()
    clean_numel = numel - n_outliers

    clean = dequantize_4bit(
        q, s, z, torch.Size([clean_numel]), group_size=group_size
    ).flatten().to(torch.float32)

    x_flat = torch.empty(numel, dtype=torch.float32)
    mask = torch.ones(numel, dtype=torch.bool)
    mask[outlier_idx] = False
    x_flat[mask] = clean
    x_flat[outlier_idx] = outlier_val.float()
    return x_flat.view(shape).to(torch.float16)
