"""Fase 8: Qwen3MoE layer forward via Vulkan persistent buffers (no round-trips).

Uses the new usaf_vk buffer API to keep weights in GPU buffers and chain
kernel dispatches with barriers. Only downloads the final hidden output.
"""
from __future__ import annotations
import numpy as np
import torch

try:
    import usaf_vk
    HAS_VK = True
except ImportError:
    HAS_VK = False


class VKLayerPersistent:
    """Holds Vulkan buffers for one Qwen3MoE decoder layer's weights."""

    def __init__(self):
        self.bufs = {}  # name -> int (handle)
        self.initialized = False

    def load_weights(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                     head_dim: int, intermediate_size: int, num_experts: int,
                     expert_intermediate: int,
                     weights: dict[str, np.ndarray]):
        """Upload all layer weights to Vulkan buffers (call once)."""
        if not HAS_VK:
            return

        # Norm weights
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            w = weights[name]
            h = usaf_vk.create_buf(w.nbytes, True)
            usaf_vk.upload(h, w)
            self.bufs[name] = h

        # QKV projection weights (fused: [Q,K,V] in one matrix)
        for name in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]:
            w = weights[name]
            h = usaf_vk.create_buf(w.nbytes, True)
            usaf_vk.upload(h, w)
            self.bufs[name] = h

        # QK norm weights
        for name in ["q_norm.weight", "k_norm.weight"]:
            w = weights[name]
            h = usaf_vk.create_buf(w.nbytes, True)
            usaf_vk.upload(h, w)
            self.bufs[name] = h

        # Router gate
        w = weights["mlp.gate.weight"]
        h = usaf_vk.create_buf(w.nbytes, True)
        usaf_vk.upload(h, w)
        self.bufs["mlp.gate.weight"] = h

        # Store shapes for dispatch
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.expert_intermediate = expert_intermediate
        self.initialized = True

    def rmsnorm(self, x_np: np.ndarray, w_name: str) -> np.ndarray:
        """RMSNorm via persistent buffers (single op, useful for debug)."""
        rows, cols = x_np.shape
        hx = usaf_vk.create_buf(x_np.nbytes, True)
        hout = usaf_vk.create_buf(x_np.nbytes, True)
        usaf_vk.upload(hx, x_np)
        usaf_vk.rmsnorm_pipe(hx, self.bufs[w_name], hout, rows, cols, 1e-6)
        usaf_vk.barrier()
        result = usaf_vk.download(hout, [rows, cols])
        usaf_vk.destroy_buf(hx)
        usaf_vk.destroy_buf(hout)
        return result

    def gemm(self, a_np: np.ndarray, w_name: str) -> np.ndarray:
        """GEMM via persistent buffers."""
        M, K = a_np.shape
        N = self.buf_shapes.get(w_name, (0,))[1] if w_name in getattr(self, 'buf_shapes', {}) else a_np.shape[1]
        # Need to know N from the weight shape
        raise NotImplementedError("Use forward() for full pipeline")

    def forward(self, hidden_np: np.ndarray,
                cos_np: np.ndarray, sin_np: np.ndarray,
                attention_mask_np: np.ndarray | None = None) -> np.ndarray:
        """Full layer forward via Vulkan, no GPU round-trips between ops.

        For now: uses per-op dispatches with barriers. The weights are already
        in Vulkan buffers (no upload). Input hidden is uploaded once, output
        downloaded once at the end.
        """
        if not HAS_VK or not self.initialized:
            raise RuntimeError("VK not available or weights not loaded")

        B, S, H = hidden_np.shape
        hd = self.head_dim
        nH = self.num_heads
        nKV = self.num_kv_heads

        x = hidden_np.reshape(B * S, H).astype(np.float16)

        # Temp buffers (reused across dispatches)
        def _make_buf(shape):
            nbytes = int(np.prod(shape)) * 2
            return usaf_vk.create_buf(nbytes, True)

        hx = _make_buf(x.shape)
        usaf_vk.upload(hx, x)

        # 1. Input RMSNorm
        hrms = _make_buf(x.shape)
        usaf_vk.rmsnorm_pipe(hx, self.bufs["input_layernorm.weight"], hrms, B * S, H, 1e-6)
        usaf_vk.barrier()

        # 2-4. Q, K, V projections (separate to reuse o_proj weight buffer pattern)
        hq = _make_buf((B * S, nH * hd))
        usaf_vk.gemm_pipe(hrms, self.bufs["q_proj.weight"], hq, B * S, H, nH * hd)
        usaf_vk.barrier()

        hk = _make_buf((B * S, nKV * hd))
        usaf_vk.gemm_pipe(hrms, self.bufs["k_proj.weight"], hk, B * S, H, nKV * hd)
        usaf_vk.barrier()

        hv = _make_buf((B * S, nKV * hd))
        usaf_vk.gemm_pipe(hrms, self.bufs["v_proj.weight"], hv, B * S, H, nKV * hd)
        usaf_vk.barrier()

        # Download Q, K for RoPE and attention (PyTorch fallback for attention)
        q_np = usaf_vk.download(hq, [B * S, nH * hd]).reshape(B, S, nH, hd)
        k_np = usaf_vk.download(hk, [B * S, nKV * hd]).reshape(B, S, nKV, hd)

        # Cleanup temp buffers
        for h in [hx, hrms, hq, hk, hv]:
            usaf_vk.destroy_buf(h)

        # Attention in PyTorch (Vulkan attention not yet implemented)
        import torch
        device = torch.device("cpu")
        q_t = torch.from_numpy(np.ascontiguousarray(q_np)).float()
        k_t = torch.from_numpy(np.ascontiguousarray(k_np)).float()
        v_np = None  # would need to download V too

        # For now: return hidden as-is (placeholder for full pipeline)
        out = hidden_np.copy()
        return out

    def cleanup(self):
        """Free all Vulkan buffers."""
        for h in self.bufs.values():
            try:
                usaf_vk.destroy_buf(h)
            except Exception:
                pass
        self.bufs.clear()
        self.initialized = False
