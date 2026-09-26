"""Batched replacement for the per-expert expert loop.

The loop is 2 * num_experts separate matmuls per layer, and every one of them
takes the same hidden_states. For ZAYA that is 32 matmuls per layer, 1280 per
forward, on a batch of 128 tokens. N=128 is a poor M for a GPU, so the cost per
step is dominated by launch overhead rather than arithmetic.

What is NOT true, and this file exists because of it: bmm does not speed up the
training loop. A sparse run captures gradients through per-expert hooks on
detached 2D slices, and that capture is the whole mechanism by which a sparse
step receives any gradient at all. Keeping the capture means keeping the loop,
because the loop is what produces the slices the hooks are attached to.

So there are two paths here and only one of them is an optimisation:

  * capture active - training, the importance phase, anything that measures a
    gradient - the loop, unchanged. Neither a slowdown nor an optimisation, just
    the code that works.
  * no capture - evaluation, perplexity, generation, and the frozen-cache build
    - bmm. Those passes run the same arithmetic over every expert and keep
    nothing, which is exactly the case bmm is for.

The frozen cache is 124 forward passes over 29 layers before training starts, and
the earlier ZAYA run spent about 50 minutes there. That is where this pays, and
it pays there because nothing is being captured.

The two paths are asserted against each other in the test to fp16 precision, not
bit equality: the loop accumulates float32 expert by expert and bmm accumulates
in the output dtype, so they cannot be identical."""
import torch

from usaf.qwen3moe_dml import dml_qwen3_experts_forward
from usaf.utils import dense_router_weights


def dml_qwen3_experts_forward_batched(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor | None = None,
    top_k_weights: torch.Tensor | None = None,
    *,
    dense_weights: torch.Tensor | None = None,
    batched: bool = True,
) -> torch.Tensor:
    """Same arithmetic as the loop, with the experts done as two batched matmuls."""
    if not batched:
        return dml_qwen3_experts_forward(
            self, hidden_states, top_k_index, top_k_weights,
            dense_weights=dense_weights,
        )

    gup, dwn = self.gate_up_proj, self.down_proj
    n_exp = int(self.num_experts)
    if dense_weights is not None:
        weights = dense_weights
    else:
        weights = dense_router_weights(
            top_k_index, top_k_weights, n_exp, skip_index=n_exp,
        )

    shape = hidden_states.shape
    hs = hidden_states.reshape(-1, shape[-1])
    # [E, N, H] x [E, H, 2I] -> [E, N, 2I]
    gu = torch.bmm(hs.expand(n_exp, *hs.shape), gup.transpose(1, 2))
    gate, up = gu.chunk(2, dim=-1)
    cur = self.act_fn(gate) * up                      # [E, N, I]
    # [E, N, I] x [E, I, H] -> [E, N, H]
    out = torch.bmm(cur, dwn.transpose(1, 2))

    w = weights.t().to(out.dtype).unsqueeze(-1)        # [E, N, 1]
    final = (out * w).sum(dim=0).float()
    return final.reshape(shape).to(hidden_states.dtype)


def batched_or_loop_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor | None = None,
    top_k_weights: torch.Tensor | None = None,
    *,
    dense_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pick the path: the loop whenever a gradient is being captured."""
    capturing = (
        torch.is_grad_enabled()
        and getattr(self, "_grad_capture", None) is not None
    )
    return dml_qwen3_experts_forward_batched(
        self, hidden_states, top_k_index, top_k_weights,
        dense_weights=dense_weights,
        batched=not capturing,
    )
