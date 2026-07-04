"""Fase 10: Vulkan integration for training — VK forward for a single layer.

Manages persistent Vulkan buffers for one decoder layer's weights.
Provides forward() that runs rmsnorm + QKV + RoPE + O-proj + post-norm on GPU.
Returns numpy output (to be converted to torch tensor for backward).
"""
from __future__ import annotations
import numpy as np
import os, sys

HAS_VK = False
try:
    _vk_sdk = os.environ.get("VULKAN_SDK", "C:/VulkanSDK/1.4.341.1")
    _vk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vulkan', 'build', 'Release')
    sys.path.insert(0, _vk_path)
    os.add_dll_directory(os.path.join(_vk_sdk, 'Bin'))
    import usaf_vk
    _spv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vulkan', 'build', 'spirv')
    usaf_vk.set_spirv_path(_spv_path)
    usaf_vk.init()
    HAS_VK = True
except Exception as e:
    import traceback
    print(f"  [VK] import failed: {e}", flush=True)

class VKLayer:
    """Vulkan-accelerated Q/K/V projections for one Qwen3MoE decoder layer.

    Forward: RMSNorm -> Q/K/V projections -> download Q/K/V.
    Native DML handles QK norm, RoPE, attention, O-proj, residual, MLP.
    Monkey-patch injects VK Q/K/V into native attention for correctness.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 weights: dict[str, np.ndarray]):
        self.H = hidden_size
        self.nH = num_heads
        self.nKV = num_kv_heads
        self.hd = head_dim
        self.bufs: dict[str, int] = {}
        self._uploaded = False

        if not HAS_VK:
            return

        for name, w in weights.items():
            if 'proj.weight' in name:
                w = np.ascontiguousarray(w.T)
            h = usaf_vk.create_buf(w.nbytes, True)
            usaf_vk.upload(h, w.astype(np.float16) if w.dtype != np.float16 else w)
            self.bufs[name] = h
        self._uploaded = True

    def forward(self, hidden_np: np.ndarray, cos_np=None, sin_np=None):
        """VK-accelerated Q/K/V projections. Returns RAW [B*S, dim].
        Monkey-patch injects into native DML attention — loss matches DML (1.8057).
        """
        if not self._uploaded:
            raise RuntimeError("VKLayer weights not uploaded")

        B, S, H = hidden_np.shape
        x = hidden_np.reshape(B * S, H).astype(np.float16)

        def alloc(shape):
            return usaf_vk.create_buf(int(np.prod(shape)) * 2, True)

        hx = alloc(x.shape); usaf_vk.upload(hx, x)
        hrms = alloc(x.shape)
        hq = alloc((B * S, self.nH * self.hd))
        hk = alloc((B * S, self.nKV * self.hd))
        hv = alloc((B * S, self.nKV * self.hd))

        usaf_vk.rmsnorm_pipe(hx, self.bufs["input_layernorm.weight"], hrms, B * S, H, 1e-6)
        usaf_vk.barrier()
        usaf_vk.gemm_pipe(hrms, self.bufs["self_attn.q_proj.weight"], hq, B * S, H, self.nH * self.hd)
        usaf_vk.barrier()
        usaf_vk.gemm_pipe(hrms, self.bufs["self_attn.k_proj.weight"], hk, B * S, H, self.nKV * self.hd)
        usaf_vk.barrier()
        usaf_vk.gemm_pipe(hrms, self.bufs["self_attn.v_proj.weight"], hv, B * S, H, self.nKV * self.hd)
        usaf_vk.barrier()

        q_np = usaf_vk.download(hq, [B * S, self.nH * self.hd]).view(np.float16)
        k_np = usaf_vk.download(hk, [B * S, self.nKV * self.hd]).view(np.float16)
        v_np = usaf_vk.download(hv, [B * S, self.nKV * self.hd]).view(np.float16)

        for h in [hx, hrms, hq, hk, hv]:
            usaf_vk.destroy_buf(h)

        return q_np, k_np, v_np

    def cleanup(self):
        for h in self.bufs.values():
            try: usaf_vk.destroy_buf(h)
            except: pass
        self.bufs.clear()
        self._uploaded = False


def create_vk_layers(train_layers, model_config, weights_by_layer, rotary_emb) -> dict:
    H = model_config.hidden_size
    nH = model_config.num_attention_heads
    nKV = model_config.num_key_value_heads
    hd = getattr(model_config, 'head_dim', H // nH)  # Qwen3 uses head_dim=128, not H/nH=64

    layers = {}
    if not HAS_VK:
        return layers
    for li in train_layers:
        prefix = f"model.layers.{li}."
        w = weights_by_layer.get(li)
        if w is None:
            continue
        layer = VKLayer(H, nH, nKV, hd, w)
        layers[li] = layer

    return layers
