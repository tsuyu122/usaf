"""Route GraniteMoe experts through the sparse USAF forward.

GraniteMoeExperts is not one of the containers USAF replaces, so its forward
ran the dense path: every expert for every token. Nothing raises, the loss is
computed, and the run reports progress while no expert gradient is ever
captured - the training that produced "Active: 0/0" on ZAYA1, reached here by
a different route.

The forward it needs is the one the Qwen3-MoE patch installs. The signature is
already the same, (hidden_states, top_k_index, top_k_weights), and so are the
weights it reads: gate_up_proj and down_proj, both stacked over the expert
dimension, and num_experts. Granite was renamed onto the fused layout rather
than onto a different one, which is why this is a class swap and not a
translation.

Only the expert container is replaced. The router is untouched, because USAF
trains the router as an ordinary parameter rather than through this path.
"""

import torch

_ORIGINAL = None


def patch_granite_for_dml() -> bool:
    """Install the sparse expert forward on GraniteMoeExperts.

    Returns whether anything was patched, so a transformers version without
    the class is a no-op rather than an ImportError at the top of a run.
    """
    global _ORIGINAL
    try:
        from transformers.models.granitemoe import modeling_granitemoe as G
    except ImportError:
        return False

    if getattr(G.GraniteMoeExperts.forward, "_usaf_sparse", False):
        return True

    from usaf.qwen3moe_dml import dml_qwen3_experts_forward

    _ORIGINAL = G.GraniteMoeExperts.forward

    def sparse_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return dml_qwen3_experts_forward(
            self, hidden_states, top_k_index, top_k_weights)

    sparse_forward._usaf_sparse = True
    G.GraniteMoeExperts.forward = sparse_forward
    return True


def unpatch_granite_for_dml() -> None:
    """Put the original GraniteMoeExperts forward back."""
    global _ORIGINAL
    if _ORIGINAL is None:
        return
    from transformers.models.granitemoe import modeling_granitemoe as G

    G.GraniteMoeExperts.forward = _ORIGINAL

_ORIGINAL = None
