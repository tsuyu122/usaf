"""DML-safe forward replacements for Qwen3MoE MoE operations.

Avoids scatter-based ops (one_hot, index_add_, nonzero) that trigger DML
"partially modified dimensions" errors by using dense masked computation.

Backward strategy: autograd must NEVER touch the fused 3D expert tensors
([128,1536,2048] / [128,2048,768]). Selecting a slice inside the graph makes
select-backward allocate a full-size grad (>800MB contiguous), which exceeds
DML's per-allocation limit under disable_tiled_resources -> "unknown error".
Instead each expert's 2D slice is detached into a fresh leaf (grad 6-13MB)
and grads are streamed to CPU buffers via tensor hooks.

Grad capture protocol: the training script sets on the experts module
    module._grad_capture = (store: dict, prefix: str)
where `store` maps full parameter names (f"{prefix}.gate_up_proj") to CPU
fp16 buffers shaped like the fused tensors. Buffers are allocated lazily on
first hook fire and accumulate grads (including any loss scale applied by
the caller).
"""
import torch
import torch.nn.functional as F


def dml_qwen3_experts_forward(self, hidden_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Dense-masked expert loop over detached per-expert 2D slices."""
    final = torch.zeros(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
    weights_t = weights.t().contiguous()
    gup, dwn = self.gate_up_proj, self.down_proj

    capture = torch.is_grad_enabled() and getattr(self, "_grad_capture", None) is not None
    if capture:
        store, prefix = self._grad_capture

        if isinstance(store, dict):
            def _make_hook(full_name: str, full_shape, ei: int):
                def _hook(g: torch.Tensor) -> None:
                    buf = store.get(full_name)
                    if buf is None:
                        buf = torch.zeros(full_shape, dtype=torch.float16, device="cpu")
                        store[full_name] = buf
                    buf[ei] += g.detach().to(device="cpu", dtype=torch.float16)
                return _hook
        else:
            def _make_hook(full_name: str, full_shape, ei: int):
                def _hook(g: torch.Tensor) -> None:
                    store.add(full_name, ei, g)
                return _hook

    for ei in range(self.num_experts):
        wg = gup[ei].detach()
        wd = dwn[ei].detach()
        if capture:
            wg.requires_grad_(True)
            wd.requires_grad_(True)
            wg.register_hook(_make_hook(prefix + ".gate_up_proj", gup.shape, ei))
            wd.register_hook(_make_hook(prefix + ".down_proj", dwn.shape, ei))

        gu = F.linear(hidden_states, wg)
        gate, up = gu.chunk(2, dim=-1)
        cur = self.act_fn(gate) * up
        cur = F.linear(cur, wd)
        final = final + cur.float() * weights_t[ei].float().unsqueeze(-1)

    return final.to(hidden_states.dtype)


def dml_qwen3_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """MoE block without topk in the gradient path and without one_hot."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hs = hidden_states.view(-1, hidden_dim)

    router = self.gate
    router_logits = F.linear(hs, router.weight)
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)

    with torch.no_grad():
        top_val, _ = torch.topk(router_probs, router.top_k, dim=-1)
        thresh = top_val[:, -1:].clone()
        mask = (router_probs >= thresh).to(router_probs.dtype)
    weights = router_probs * mask
    if router.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    weights = weights.to(hidden_states.dtype)

    final = self.experts(hs, weights)
    return final.view(batch_size, sequence_length, hidden_dim)


def patch_qwen3moe_for_dml():
    """Applies DML-safe forwards (affects all model instances)."""
    from transformers.models.qwen3_moe import modeling_qwen3_moe
    modeling_qwen3_moe.Qwen3MoeExperts.forward = dml_qwen3_experts_forward
    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward = dml_qwen3_moe_block_forward
    return modeling_qwen3_moe
