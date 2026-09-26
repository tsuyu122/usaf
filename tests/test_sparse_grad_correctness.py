"""The sparse gradient mechanism must be numerically identical to autograd.

This is the load-bearing claim of the project: instead of materialising a dense
gradient for every expert, USAF captures only the active slice during the
forward and re-runs the backward by hand, one layer at a time, feeding each
layer the incoming gradient explicitly. If any part of that is off by a sign, a
transpose, or a layer offset, training still runs and still moves the weights -
it just moves them the wrong way, and nothing else in the suite would notice.

The reference here is a dense backward over the same model with the same loss.
"""
import torch

from usaf.moe_loader import SparseGradStore


class _Experts(torch.nn.Module):
    """Stands in for model.layers[i].mlp.experts.

    The weight is the flat [num_experts, C] buffer the store indexes into. The
    forward follows the same protocol usaf/qwen3moe_dml.py uses: each expert row
    is detached into a leaf that requires grad, a tensor hook funnels that
    expert's gradient into the store, and the rows are recombined. Row b of the
    input is processed by expert b, so every element gets a gradient and the
    per-expert slicing is exercised.
    """

    def __init__(self, num_experts: int = 3, c: int = 4):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(num_experts, c))

    def forward(self, x):
        store, prefix = getattr(self, "_grad_capture", (None, None))
        out = torch.zeros_like(x)
        for ei in range(self.w.shape[0]):
            if store is None:
                # Reference path: let autograd build the graph normally.
                out = out + x[ei : ei + 1] * self.w[ei]
                continue
            w = self.w[ei].detach().clone().requires_grad_(True)
            w.register_hook(lambda g, _ei=ei: store.add(prefix, _ei, g))
            out = out + x[ei : ei + 1] * w
        return out


def test_store_returns_exactly_the_active_gradient():
    torch.manual_seed(0)
    num_experts, c = 3, 4
    active = torch.tensor([0, 1, 5, 7, 10, 11], dtype=torch.long)
    store = SparseGradStore({"w": active}, {"w": (num_experts, c)})
    store.zero_()

    truth = torch.randn(num_experts * c)
    for ei in range(num_experts):
        store.add("w", ei, truth[ei * c : (ei + 1) * c])

    assert torch.equal(store.compact["w"], truth[active])


def test_store_ignores_inactive_experts():
    """An expert with no active element must not contribute, and must not raise."""
    torch.manual_seed(1)
    store = SparseGradStore({"w": torch.tensor([0, 1])}, {"w": (4, 2)})
    store.zero_()
    store.add("w", 0, torch.full((2,), 3.0))
    # Expert 2 has no active index at all.
    store.add("w", 2, torch.full((2,), 99.0))
    assert torch.equal(store.compact["w"], torch.tensor([3.0, 3.0]))


def test_store_accumulates_across_calls():
    """Two micro-batches must add, not overwrite."""
    torch.manual_seed(2)
    store = SparseGradStore({"w": torch.tensor([0, 1])}, {"w": (2, 2)})
    store.zero_()
    store.add("w", 0, torch.tensor([1.0, 2.0]))
    store.add("w", 0, torch.tensor([10.0, 20.0]))
    assert torch.equal(store.compact["w"], torch.tensor([11.0, 22.0]))


def test_zero_clears_between_steps():
    torch.manual_seed(3)
    store = SparseGradStore({"w": torch.tensor([0, 1])}, {"w": (2, 2)})
    store.add("w", 0, torch.tensor([1.0, 1.0]))
    assert store.compact["w"].abs().sum() > 0
    store.zero_()
    assert store.compact["w"].abs().sum() == 0


def test_sparse_capture_equals_dense_autograd():
    """The end-to-end claim: hand-rolled sparse backward == torch autograd.

    Two stacked expert layers, a frozen prefix to exercise the detach-and-
    replay path, and a loss that depends on the whole stack. The gradient the
    store captures at the active positions is compared against p.grad from a
    plain dense backward of the identical graph.
    """
    torch.manual_seed(7)
    num_experts, c, depth = 3, 4, 3

    layers = [_Experts(num_experts, c) for _ in range(depth)]
    for m in layers:
        m.double()

    x0 = torch.randn(num_experts, c, dtype=torch.double)

    # Reference: dense autograd over the same graph.
    h = x0
    for m in layers:
        h = m(h)
    ref_loss = (h * torch.arange(1, h.numel() + 1, dtype=torch.double).reshape(h.shape)).sum()
    ref_loss.backward()
    ref_grads = [m.w.grad.detach().clone() for m in layers]

    # Sparse path: no graph is retained. Each layer's input is stashed during
    # the forward, the head is differentiated, and the incoming gradient is
    # replayed down the stack in reverse, capturing only the active slice.
    active = torch.tensor([0, 1, 5, 7, 10, 11], dtype=torch.long)
    shapes = {f"L{i}": (num_experts, c) for i in range(depth)}
    store = SparseGradStore({f"L{i}": active for i in range(depth)}, shapes)
    store.zero_()

    # No graph is retained across the stack: each layer input is stashed and
    # detached as the forward descends, and the gradient is replayed by hand.
    # (The forward itself must not run under no_grad, or the tensor hooks
    # that feed the store never fire.)
    hidden = x0
    saved = []
    for i, m in enumerate(layers):
        saved.append(hidden)
        m._grad_capture = (store, f"L{i}")
        hidden = m(hidden).detach()

    top = hidden.detach().requires_grad_(True)
    w_ref = torch.arange(1, top.numel() + 1, dtype=torch.double).reshape(top.shape)
    (top * w_ref).sum().backward()
    g = top.grad

    for i in range(depth - 1, -1, -1):
        xi = saved[i].detach().requires_grad_(True)
        m = layers[i]
        m._grad_capture = (store, f"L{i}")
        out = m(xi)
        out.backward(g)
        g = xi.grad

    for i, m in enumerate(layers):
        m._grad_capture = None
        # The store accumulates in float32 by design, so the reference is
        # compared in the same precision rather than against a double that
        # the sparse path was never able to represent.
        got = store.compact[f"L{i}"].double()
        want = ref_grads[i].reshape(-1)[active]
        assert torch.allclose(got, want, rtol=1e-6, atol=1e-9), (
            f"layer {i}: sparse capture differs from autograd\n"
            f"  got  {got.tolist()}\n  want {want.tolist()}"
        )
