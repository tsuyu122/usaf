"""DML-safe forward replacements for OLMoE MoE operations.

Avoids scatter-based ops (one_hot, topk, index_add/index_select) that trigger
DML "partially modified dimensions" errors by using dense masked computation.
"""
import torch
import torch.nn.functional as F


def dml_experts_forward(self, hidden_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Dense-masked experts forward. weights: [N, E] (router weight, 0 outside top-k).

    CRITICAL (fp16 overflow): Each expert computes output for ALL tokens and
    multiplies by weight w (0 outside top-k). With real weights (trained in bf16),
    raw expert output can overflow fp16 (max 65504) -> inf; inf * 0 = NaN, which
    contaminates logits and zeroes gradients (loss becomes ln(vocab)). bf16 does
    not overflow, fp16 does. Solution: run expert math in fp32 (huge range, no
    overflow). Tensors here are small ([N, hidden], N=tokens), so fp32 cost is low.
    """
    hs32 = hidden_states.float()
    final = torch.zeros_like(hs32)
    weights_t = weights.t().contiguous()

    for expert_idx in range(self.num_experts):
        w = weights_t[expert_idx].float()
        gu = F.linear(hs32, self.gate_up_proj[expert_idx].float())
        gate, up = gu.chunk(2, dim=-1)
        current = self.act_fn(gate) * up
        current = F.linear(current, self.down_proj[expert_idx].float())
        current = current * w.unsqueeze(-1)
        final += current

    return final.to(hidden_states.dtype)


def dml_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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


def patch_olmoe_for_dml():
    """Applies DML-safe forwards (affects all model instances)."""
    from transformers.models.olmoe import modeling_olmoe
    modeling_olmoe.OlmoeExperts.forward = dml_experts_forward
    modeling_olmoe.OlmoeSparseMoeBlock.forward = dml_moe_block_forward
    return modeling_olmoe
