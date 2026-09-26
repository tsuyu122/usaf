"""Coverage for the paths that shipped with no test at all.

A coverage sweep of the package found large untested regions - moe_loader
27/40 definitions, vk_layer 8/10, qwen3_layer_vk 11/15, data 8/11,
model_factory 8/10 - which is how four dead CLI flags, an inverted selector and
an export that drifted untouched weights all survived. These pin the parts of
that surface that are reachable without a GPU.
"""
import os

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = r"C:\Users\hm\Projects\e2e"


# --- TopKImportanceStore: the RigL selection that feeds every training step ---

def _imp_store(frac, experts=4, per=200, seed=0):
    from usaf.moe_loader import TopKImportanceStore

    torch.manual_seed(seed)
    shapes = {"layer0.gate_up": (experts, per)}
    imp = TopKImportanceStore(shapes, frac=frac)
    for ei in range(experts):
        imp.add("layer0.gate_up", ei, torch.randn(per))
    return imp, imp.select()["layer0.gate_up"], experts * per


@pytest.mark.parametrize("frac", [0.01, 0.02, 0.1, 0.5])
def test_importance_store_selects_exactly_the_requested_fraction(frac):
    _imp, idx, total = _imp_store(frac)
    assert idx.numel() == int(total * frac), (
        f"frac={frac} selected {idx.numel()} of {total}"
    )


def test_importance_indices_are_unique_and_in_range():
    """Expert e occupies [e*P, (e+1)*P), so the per-expert candidate blocks
    cannot overlap - but only if the global offset is applied, and a missing
    offset would make every expert contribute the same low indices.
    """
    _imp, idx, total = _imp_store(0.1)
    assert len(set(idx.tolist())) == idx.numel(), "duplicate active indices"
    assert bool((idx >= 0).all() and (idx < total).all())


def test_importance_picks_the_largest_gradients():
    """A selection that did not track magnitude would still return the right
    count, so the count alone cannot be trusted.

    add() keeps cand_mult x frac candidates per expert before select() merges
    them, so the slice needs distinct values or the tie among the untouched
    zeros fills the quota.
    """
    from usaf.moe_loader import TopKImportanceStore

    imp = TopKImportanceStore({"m": (1, 10)}, frac=0.3)
    # candidates = top 10*0.3*2 = 6, selected = top 3 = values 10, 9, 8
    imp.add("m", 0, torch.arange(1.0, 11.0))
    picked = set(imp.select()["m"].tolist())
    assert picked == {7, 8, 9}, f"picked {sorted(picked)}, expected the three largest"


def test_zero_clears_previous_candidates():
    from usaf.moe_loader import TopKImportanceStore

    imp = TopKImportanceStore({"m": (1, 100)}, frac=0.1)
    imp.add("m", 0, torch.randn(100))
    assert imp.select()
    imp.zero_()
    assert imp.select() == {}, "zero_() left candidates behind"


# --- SparseGradStore / SparseAdam ---

def test_sparse_grad_store_compacts_to_the_active_positions():
    """Only the active elements of each expert slice are kept.

    shapes is (num_experts, per_expert): the store derives the per-expert slice
    size from shape[1:] to map a global flat index back to (expert, local). A
    one-element shape makes every index its own expert, which silently drops
    the gradient rather than raising.
    """
    from usaf.moe_loader import SparseGradStore

    n_experts, per_expert = 2, 500
    idx = torch.arange(0, n_experts * per_expert, 50)
    st = SparseGradStore(active_idx={"m": idx}, shapes={"m": (n_experts, per_expert)})
    assert st.compact["m"].numel() == idx.numel()

    st.zero_()
    full = torch.zeros(n_experts * per_expert)
    full[idx] = 7.0
    for ei in range(n_experts):
        st.add("m", ei, full[ei * per_expert:(ei + 1) * per_expert])

    comp = st.compact["m"]
    assert comp.numel() == idx.numel()
    assert torch.allclose(comp, torch.full((idx.numel(),), 7.0), atol=1e-3), (
        f"compact mean {comp.mean():.3f}, expected all 7.0"
    )


def test_sparse_grad_store_ignores_inactive_positions():
    """A gradient outside the active set must not reach the compact vector."""
    from usaf.moe_loader import SparseGradStore

    n_experts, per_expert = 2, 500
    idx = torch.arange(0, 10 * 50, 50)
    st = SparseGradStore(active_idx={"m": idx}, shapes={"m": (n_experts, per_expert)})
    st.zero_()
    full = torch.zeros(n_experts * per_expert)
    full[1] = 99.0
    for ei in range(n_experts):
        st.add("m", ei, full[ei * per_expert:(ei + 1) * per_expert])
    assert float(st.compact["m"].abs().max()) == 0.0

# --- quantization helpers that shipped with no test ---

def test_reconstruction_error_is_relative_to_the_signal_not_absolute():
    from usaf.quantization import dequantize_4bit, quantize_4bit, reconstruction_error

    torch.manual_seed(0)
    w = (torch.randn(512) * 0.05).half()
    q, s, z, shape = quantize_4bit(w, 128)
    deq = dequantize_4bit(q, s, z, shape)
    err = reconstruction_error(w, deq)

    assert 0.0 < err < 1.0, f"relative MSE should be a small positive ratio, got {err}"

    scaled = reconstruction_error(w * 10, deq * 10)
    assert abs(scaled - err) < 1e-6, "the metric is not scale invariant"


def test_quantization_error_shrinks_as_more_outliers_are_kept():
    """The outlier path exists to reduce error. If it did not, the extra
    machinery would be pointless, and a count-only test would not notice.
    """
    from usaf.quantization import dequantize_with_outliers, quantize_with_outliers, reconstruction_error

    torch.manual_seed(1)
    w = (torch.randn(1024) * 0.05).half()

    errs = []
    for frac in (0.0, 0.05, 0.5):
        packed = quantize_with_outliers(w, 128, frac)
        deq = dequantize_with_outliers(packed)
        errs.append(reconstruction_error(w, deq.reshape(w.shape).to(w.dtype)))

    assert errs[0] > errs[1] > errs[2], f"error did not fall: {errs}"


def test_outlier_packing_keeps_the_shape_and_the_outlier_count():
    from usaf.quantization import quantize_with_outliers

    torch.manual_seed(2)
    w = (torch.randn(1024) * 0.05).half()
    packed = quantize_with_outliers(w, 128, 0.05)
    n = packed["outlier_indices"].numel()
    assert n == int(1024 * 0.05), f"{n} outliers for a 5% fraction"
    assert len(set(packed["outlier_indices"].tolist())) == n, "duplicate outliers"


def test_state_dict_size_estimate_reports_real_compression():
    from usaf.quantization import estimate_quantized_state_dict_size

    w = (torch.randn(4096) * 0.05).half()
    out = estimate_quantized_state_dict_size({"a": w})
    ratio = out["compression_ratio"]
    assert 2.0 < ratio < 5.0, f"4-bit with fp16 scales should compress 2-5x, got {ratio}"


def test_estimate_quantized_size_grows_with_the_parameter_count():
    from usaf.quantization import estimate_quantized_size

    small = estimate_quantized_size(1000)["total_mb"]
    large = estimate_quantized_size(1_000_000)["total_mb"]
    assert large > small * 100


# --- data pipeline ---

def test_cpp_dataset_and_dataloader_produce_batched_tensors():
    from usaf.data import CppDataset, create_dataloader

    samples = [{"input_ids": list(range(8)), "labels": list(range(8))} for _ in range(4)]
    ds = CppDataset(samples)
    assert len(ds) == 4

    batch = next(iter(create_dataloader(ds, batch_size=2, shuffle=False)))
    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_ids"].dtype == torch.long


def test_dataloader_shuffle_does_not_lose_or_duplicate_samples():
    from usaf.data import CppDataset, create_dataloader

    samples = [{"input_ids": [i] * 4, "labels": [i] * 4} for i in range(8)]
    ds = CppDataset(samples)
    seen = []
    """drop_last was hardcoded True, so a dataset that is not a multiple of the
    batch size silently lost up to batch_size-1 samples every pass, and the
    caller could not turn it off. On 8 samples with batch_size 3 that is 25% of
    the data gone.
    """
    for b in create_dataloader(ds, batch_size=3, shuffle=True, drop_last=False):
        seen.extend(b["input_ids"][:, 0].tolist())
    assert sorted(seen) == list(range(8)), "lost or duplicated: " + str(sorted(seen))


def test_dataloader_default_still_drops_the_partial_batch():
    """The default is unchanged, so existing runs behave identically."""
    from usaf.data import CppDataset, create_dataloader

    samples = [{"input_ids": [i] * 4, "labels": [i] * 4} for i in range(8)]
    ds = CppDataset(samples)
    n = sum(b["input_ids"].shape[0] for b in create_dataloader(ds, batch_size=3))
    assert n == 6, "expected the default to keep 6 of 8, kept " + str(n)
