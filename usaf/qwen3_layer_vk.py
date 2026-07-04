"""Qwen3MoE Decoder Layer with Vulkan-accelerated kernels.

Strategy:
- RMSNorm, GEMM (Q/K/V/O projections, MoE experts), and RoPE run in Vulkan.
- Attention softmax, topk, silu, gather/scatter run in PyTorch.
- Expert weights are dequantized on-demand via Vulkan kernel.

Full forward matches the HF golden_capture.py reference (dense MoE for validation).
"""

import os, sys
import numpy as np
import torch
import torch.nn.functional as F

# Vulkan Python bindings — binary is in usaf/vulkan/build/Release/
_vk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "vulkan", "build", "Release"))
if _vk_path not in sys.path:
    sys.path.insert(0, _vk_path)
# Ensure Vulkan runtime DLL is findable
_vk_sdk_bin = os.environ.get("VULKAN_SDK", "")
if _vk_sdk_bin:
    _vk_dll = os.path.join(_vk_sdk_bin, "Bin")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_vk_dll)
        except Exception:
            pass
try:
    import usaf_vk
    HAS_VK = True
except ImportError:
    HAS_VK = False
    print("[WARN] usaf_vk not available, falling back to PyTorch for all ops")

if HAS_VK:
    _spirv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "vulkan", "build", "spirv"))
    usaf_vk.set_spirv_path(_spirv_path)
    usaf_vk.init()


def _to_uint16(x: torch.Tensor) -> np.ndarray:
    """Convert fp16 tensor to uint16 numpy (raw bits) for Vulkan."""
    return x.contiguous().view(torch.uint16).cpu().numpy()


def _from_uint16(arr: np.ndarray, shape) -> torch.Tensor:
    """Convert uint16 numpy back to fp16 torch tensor."""
    return torch.from_numpy(arr).view(torch.float16).reshape(shape)


def _fp16_to(t: torch.Tensor, device, dtype=None) -> torch.Tensor:
    """Ensure tensor is fp16 on the right device."""
    if dtype is None:
        dtype = t.dtype
    if t.dtype != torch.float16:
        t = t.to(torch.float16)
    if t.device != device:
        t = t.to(device)
    return t


def _ensure_shape(t: torch.Tensor, shape, device) -> torch.Tensor:
    """Reshape if needed, ensure contiguous fp16 on device."""
    t = t.reshape(shape).contiguous()
    if t.dtype != torch.float16:
        t = t.to(torch.float16)
    if t.device != device:
        t = t.to(device)
    return t


def rmsnorm_vk(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Vulkan RMSNorm: x [rows, cols], w [cols] -> [rows, cols]."""
    if not HAS_VK:
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * w
    rows, cols = x.shape
    y_np = usaf_vk.rmsnorm(_to_uint16(x), _to_uint16(w), rows, cols, eps)
    return _from_uint16(y_np, (rows, cols)).to(x.device)


def gemm_vk(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vulkan GEMM: a[M,K] @ b[K,N] -> [M,N]."""
    if not HAS_VK:
        return torch.matmul(a.float(), b.float()).to(a.dtype)
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, f"GEMM shape mismatch: {a.shape} @ {b.shape}"
    c_np = usaf_vk.gemm(_to_uint16(a), _to_uint16(b), M, K, N)
    return _from_uint16(c_np, (M, N)).to(a.device)


def rope_vk(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Vulkan RoPE: q[B,nH,S,hd], k[B,nKV,S,hd], cos[S,hd], sin[S,hd]."""
    if not HAS_VK:
        def _rope(x, c, s):
            xf = x.float()
            x1, x2 = xf.chunk(2, dim=-1)          # rotate_half (half-split, Qwen3/HF)
            xr = torch.cat((-x2, x1), dim=-1)
            c4 = c.unsqueeze(0).unsqueeze(0)
            s4 = s.unsqueeze(0).unsqueeze(0)
            return (xf * c4 + xr * s4).to(x.dtype)
        return _rope(q, cos, sin), _rope(k, cos, sin)

    B, nH, S, hd = q.shape
    _, nKV, _, _ = k.shape
    q_np, k_np = usaf_vk.rope(
        _to_uint16(q), _to_uint16(k),
        _to_uint16(cos), _to_uint16(sin),
        B, nH, nKV, S, hd)
    q_out = _from_uint16(q_np, (B, nH, S, hd)).to(q.device)
    k_out = _from_uint16(k_np, (B, nKV, S, hd)).to(k.device)
    return q_out, k_out


class Qwen3LayerWeights:
    """Holds layer weights as fp16 tensors on CPU, uploaded to Vulkan on demand."""
    def __init__(self, weights_dict, device="cpu"):
        self.W = {}
        for k, v in weights_dict.items():
            self.W[k] = _fp16_to(v, device)
        self.device = device

    def get(self, key):
        return self.W[key]

    @property
    def hidden_size(self):
        return self.W["input_layernorm.weight"].shape[0]

    @property
    def num_heads(self):
        w = self.W["self_attn.q_proj.weight"]
        return w.shape[0] // self.head_dim

    @property
    def num_kv_heads(self):
        w = self.W["self_attn.k_proj.weight"]
        return w.shape[0] // self.head_dim

    @property
    def head_dim(self):
        return 128  # Qwen3-30B-A3B


def qwen3_layer_forward_vk(
    hidden: torch.Tensor,           # [B, S, H] fp16
    W: Qwen3LayerWeights,
    cos: torch.Tensor,              # [S, hd] fp16, interleaved
    sin: torch.Tensor,
    mask: torch.Tensor,             # [1, 1, S, S] or None
    expert_gate_up: torch.Tensor,   # [n_exp, 2*inter, H] fp16 (dequantized)
    expert_down: torch.Tensor,      # [n_exp, H, inter] fp16
    eps: float = 1e-6,
    top_k: int = 8,
):
    """Full Qwen3MoE decoder layer forward, Vulkan-accelerated.
    Matches HF golden_capture.py reference (dense MoE for validation).
    """
    B, S, H = hidden.shape
    nH = W.num_heads
    nKV = W.num_kv_heads
    hd = W.head_dim
    n_exp = expert_gate_up.shape[0]
    inter = expert_gate_up.shape[1] // 2
    device = hidden.device

    # 1. Input RMSNorm
    hs = hidden.reshape(-1, H)  # [B*S, H]
    w_ln = W.get("input_layernorm.weight")
    x = rmsnorm_vk(hs, w_ln, eps)  # [B*S, H]

    # 2. Q/K/V projections (GEMM)
    q = gemm_vk(x, W.get("self_attn.q_proj.weight").T).reshape(B, S, nH, hd)
    k = gemm_vk(x, W.get("self_attn.k_proj.weight").T).reshape(B, S, nKV, hd)
    v = gemm_vk(x, W.get("self_attn.v_proj.weight").T).reshape(B, S, nKV, hd)

    # QK Norm (RMSNorm per head, over head_dim)
    q = q.reshape(B * S * nH, hd)
    q = rmsnorm_vk(q, W.get("self_attn.q_norm.weight"), eps)
    q = q.reshape(B, S, nH, hd).transpose(1, 2)  # [B, nH, S, hd]
    k = k.reshape(B * S * nKV, hd)
    k = rmsnorm_vk(k, W.get("self_attn.k_norm.weight"), eps)
    k = k.reshape(B, S, nKV, hd).transpose(1, 2)
    v = v.reshape(B, S, nKV, hd).transpose(1, 2)  # [B, nKV, S, hd]

    # 3. RoPE
    q, k = rope_vk(q, k, cos, sin)  # Vulkan RoPE

    # 4. Attention (PyTorch)
    n_rep = nH // nKV
    k_rep = k.repeat_interleave(n_rep, dim=1)
    v_rep = v.repeat_interleave(n_rep, dim=1)
    scaling = hd ** -0.5
    scores = torch.matmul(q, k_rep.transpose(2, 3)) * scaling
    if mask is not None:
        scores = scores + mask[:, :, :S, :S]
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    attn = torch.matmul(probs, v_rep)
    attn = attn.transpose(1, 2).reshape(B, S, nH * hd)

    # 5. O projection
    attn = gemm_vk(attn.reshape(-1, nH * hd), W.get("self_attn.o_proj.weight").T).reshape(B, S, H)

    # 6. Residual + Post-attention RMSNorm
    h2 = hidden + attn
    hs2 = h2.reshape(-1, H)
    x2 = rmsnorm_vk(hs2, W.get("post_attention_layernorm.weight"), eps)

    # 7. Router
    router_logits = gemm_vk(x2, W.get("mlp.gate.weight").T)  # [B*S, n_exp]
    router_probs = F.softmax(router_logits.float(), dim=-1)
    tv, ti = torch.topk(router_probs, top_k, dim=-1)

    # Normalize top-k weights (HF convention)
    tv = tv / tv.sum(-1, keepdim=True)

    # 8. MoE (sparse dispatch: only process tokens assigned to each expert)
    moe = torch.zeros_like(x2)  # [B*S, H]
    n_tokens = B * S

    # Build per-expert token lists from topk indices
    for ei in range(n_exp):
        # Which tokens selected this expert, and at which topk position?
        mask = (ti == ei)  # [n_tokens, top_k] bool
        token_indices, k_positions = mask.nonzero(as_tuple=True)
        if token_indices.numel() == 0:
            continue

        # Gather tokens assigned to this expert
        x_e = x2[token_indices]  # [n_assigned, H]
        w_e = tv[token_indices, k_positions].float()  # [n_assigned], router weights

        # gate_up_proj: [n_assigned, H] @ [H, 2*inter] -> [n_assigned, 2*inter]
        gu = gemm_vk(x_e, expert_gate_up[ei].T)
        g, u = gu.chunk(2, dim=-1)
        act = (F.silu(g.float()).to(g.dtype) * u)  # SwiGLU

        # down_proj: [n_assigned, inter] @ [inter, H] -> [n_assigned, H]
        cur = gemm_vk(act, expert_down[ei].T)

        # Scatter-add weighted output
        moe.index_add_(0, token_indices, cur * w_e.unsqueeze(-1).to(cur.dtype))

    out = (h2.reshape(-1, H) + moe).reshape(B, S, H)
    return out
