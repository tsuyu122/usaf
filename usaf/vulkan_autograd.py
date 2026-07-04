"""Vulkan-accelerated PyTorch autograd Functions.

Each op wraps a Vulkan kernel via usaf_vk. Forward runs on GPU,
backward falls back to PyTorch (simpler and correct for now).
"""
import torch
import torch.nn.functional as F
from .qwen3_layer_vk import _to_uint16, _from_uint16, HAS_VK

if HAS_VK:
    import usaf_vk


class RMSNormVK(torch.autograd.Function):
    """Vulkan RMSNorm: y = (x / rms(x, dim=-1, eps)) * w"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        ctx.eps = eps
        ctx.save_for_backward(x, w)
        if not HAS_VK:
            return _fallback_rmsnorm(x, w, eps)
        rows, cols = x.shape
        y_np = usaf_vk.rmsnorm(_to_uint16(x), _to_uint16(w), rows, cols, eps)
        return _from_uint16(y_np, (rows, cols)).to(x.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, w = ctx.saved_tensors
        eps = ctx.eps
        with torch.no_grad():
            return _rmsnorm_backward(grad_output, x, w, eps), None, None


class LinearVK(torch.autograd.Function):
    """Vulkan GEMM: y = x @ W.T  (same as F.linear without bias)"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, W: torch.Tensor):
        ctx.save_for_backward(x, W)
        M, K = x.shape
        N, K2 = W.shape
        if K != K2:
            raise ValueError(f"GEMM shape mismatch: x[{M},{K}] @ W[{N},{K2}].T — K != K2")
        if not HAS_VK:
            return torch.matmul(x.float(), W.T.float()).to(x.dtype)
        c_np = usaf_vk.gemm(_to_uint16(x), _to_uint16(W.T.contiguous()), M, K, N)
        return _from_uint16(c_np, (M, N)).to(x.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, W = ctx.saved_tensors
        # grad_x = grad_output @ W
        grad_x = torch.matmul(grad_output.float(), W.float()).to(x.dtype)
        # grad_W = grad_output.T @ x
        grad_W = torch.matmul(grad_output.float().T, x.float()).to(W.dtype)
        return grad_x, grad_W


class RoPEVK(torch.autograd.Function):
    """Vulkan RoPE for Q and K tensors."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        ctx.save_for_backward(q, k, cos, sin)
        if not HAS_VK:
            q_out, k_out = _fallback_rope(q, k, cos, sin)
            return q_out, k_out
        B, nH, S, hd = q.shape
        _, nKV, _, _ = k.shape
        q_np, k_np = usaf_vk.rope(
            _to_uint16(q), _to_uint16(k),
            _to_uint16(cos), _to_uint16(sin),
            B, nH, nKV, S, hd)
        q_out = _from_uint16(q_np, (B, nH, S, hd)).to(q.device)
        k_out = _from_uint16(k_np, (B, nKV, S, hd)).to(k.device)
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):
        # RoPE backward: rotate back by the inverse angle
        q, k, cos, sin = ctx.saved_tensors
        with torch.no_grad():
            grad_q_in = _rope_backward(grad_q, q, cos, sin)
            grad_k_in = _rope_backward(grad_k, k, cos, sin)
        return grad_q_in, grad_k_in, None, None


# ════════════════════════════ fallback implementations ════════════════════════════

def _fallback_rmsnorm(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(x.dtype) * w


def _fallback_rope(q, k, cos, sin):
    def _rope(x, c, s):
        xf = x.float()
        xr = torch.cat([-xf[..., 1::2], xf[..., ::2]], dim=-1)
        c4 = c.unsqueeze(0).unsqueeze(0) if c.dim() == 2 else c.unsqueeze(0)
        s4 = s.unsqueeze(0).unsqueeze(0) if s.dim() == 2 else s.unsqueeze(0)
        return (xf * c4 + xr * s4).to(x.dtype)
    return _rope(q, cos, sin), _rope(k, cos, sin)


def _rmsnorm_backward(grad_output, x, w, eps):
    xf = x.float()
    gf = grad_output.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    # d/dx [(x * rstd) * w] = w * rstd * (I - (x*x^T)*rstd^2 / N)
    # Simplified: grad_x = w * rstd * (grad_output - (x * rstd) * (grad_output * x * rstd).mean(-1, keepdim=True))
    x_norm = xf * rstd
    g_norm = gf * w.float()
    grad_x = rstd * w.float() * (gf - x_norm * (g_norm * x_norm).mean(-1, keepdim=True))
    return grad_x.to(grad_output.dtype)


def _rope_backward(grad, x, cos, sin):
    """Inverse RoPE: rotate back. Since RoPE is orthogonal, backward = applying RoPE with negated sin."""
    xf = x.float()
    gf = grad.float()
    c = cos.unsqueeze(0).unsqueeze(0) if cos.dim() == 2 else cos.unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0) if sin.dim() == 2 else sin.unsqueeze(0)
    # Inverse rotation: a' = a*c + b*s, b' = b*c - a*s (negate s)
    gf_r = torch.cat([gf[..., 1::2], -gf[..., ::2]], dim=-1)
    return (gf * c + gf_r * (-s)).to(grad.dtype)
