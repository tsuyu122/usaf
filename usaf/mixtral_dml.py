"""DML-safe forward replacements for Mixtral MoE operations.

Mirrors usaf/qwen3moe_dml.py: avoids scatter-based ops (one_hot, index_add_,
nonzero) that trigger DML "partially modified dimensions" errors, and routes
the sparse expert gradients into the capture store via the same
module._grad_capture protocol.

Grad capture protocol: the training script sets on the experts module
    module._grad_capture = (store, prefix)
where store maps full parameter names (f"{prefix}.gate_up_proj") to the
sparse gradient store. Each expert 2D slice is detached into a fresh leaf so
autograd never tries to allocate a full-size 3D grad, and the leaf grads are
streamed to the store via tensor hooks.

In transformers >= 4.53 Mixtral uses the same fused expert layout as
Qwen3-MoE (MixtralExperts with gate_up_proj/down_proj under .mlp.experts),
so the parameter names produced by quantize_4bit are identical.
"""
import torch
import torch.nn.functional as F

from usaf.utils import dense_router_weights


def dml_mixtral_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor | None = None,
    top_k_weights: torch.Tensor | None = None,
    *,
    dense_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense-masked expert loop over detached per-expert 2D slices.

    Runs the expert math in fp32: each expert computes an output for every
    token and multiplies by its routing weight (0 outside top-k), so fp16
    expert outputs can overflow to inf, and inf * 0 produces NaN that
    contaminates the logits and zeroes the gradients.
    """
    hs32 = hidden_states.float()
    final = torch.zeros_like(hs32)
    if dense_weights is not None:
        weights = dense_weights
    else:
        weights = dense_router_weights(
            top_k_index, top_k_weights, self.num_experts,
            skip_index=getattr(self, "num_experts", None),
        )
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

        # Run the expert math in fp32: dense-masked evaluation computes an
        # output for every token, and fp16 expert outputs can overflow to inf;
        # inf * 0 (weight outside top-k) then yields NaN that contaminates the
        # logits and zeroes the gradients. Cast the fp16 weights to match.
        gu = F.linear(hs32, wg.float())
        gate, up = gu.chunk(2, dim=-1)
        cur = self.act_fn(gate) * up
        cur = F.linear(cur, wd.float())
        final = final + cur * weights_t[ei].unsqueeze(-1)

    return final.to(hidden_states.dtype)


def dml_mixtral_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """MoE block without topk/one_hot in the gradient path."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hs = hidden_states.view(-1, hidden_dim)

    router = self.gate
    router_logits = F.linear(hs, router.weight)
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)

    with torch.no_grad():
        top_val, _ = torch.topk(router_probs, self.top_k, dim=-1)
        thresh = top_val[:, -1:].clone()
        mask = (router_probs >= thresh).to(router_probs.dtype)
    weights = router_probs * mask
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    weights = weights.to(hidden_states.dtype)

    final = self.experts(hs, dense_weights=weights)
    return final.view(batch_size, sequence_length, hidden_dim)


def patch_mixtral_for_dml():
    """Applies DML-safe forwards (affects all model instances)."""
    from transformers.models.mixtral import modeling_mixtral
    modeling_mixtral.MixtralExperts.forward = dml_mixtral_experts_forward
    modeling_mixtral.MixtralSparseMoeBlock.forward = dml_mixtral_moe_block_forward
    return modeling_mixtral
