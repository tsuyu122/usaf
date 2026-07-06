# Expert dispatch — the GPU is not the bottleneck (T4)

The default MoE forward computes **all 128 experts for every token** (dense-masked).
We tried two ways to compute only the routed top-k experts, expecting a speedup:

1. **Sparse loop** — per-expert Python loop, each expert runs only on its tokens
   (`index_select` / `index_add`).
2. **Batched grouped GEMM** — group tokens by expert into a padded `[E, cap, H]`
   tensor and run all 128 experts with **2 `bmm` launches** instead of 128 GEMMs.

Same run config throughout ([cuda_t4.md](cuda_t4.md): Qwen3-30B-A3B, 1× T4, 4 trainable
layers, seq 256, FRAC 0.5 %), so the only variable is the expert-dispatch strategy.

## Result

| Strategy | Expert launches / layer | Steady step | tok/s | Probe loss |
|---|---|---|---|---|
| Dense-masked (all 128) | 128 medium GEMMs | 56.8 s | 4.5 | 1.68 → 1.19 |
| Sparse loop (top-k) | 128 tiny GEMMs | 60.0 s | 4.3 | 1.67 → 1.12 |
| **Batched grouped GEMM** | **2 `bmm`** | **55.8 s** | **4.6** | 1.67 → 1.18 |

All three are **within ~5 %** of each other, and all are **numerically identical**
(same importance losses, same 0.500 % selection, same clean loss decrease).

## The finding

Replacing 128 sequential GEMM launches with **2 batched `bmm` calls changed nothing.**
That is conclusive: **expert GEMM is not the bottleneck.** At ~14 s per trainable layer
on a T4 — a card that does these matmuls in microseconds — the step is not GPU-compute
bound at all. The time is spent **CPU-side**:

- **Per-expert gradient capture.** The capture hook ships each of the 128 experts'
  weight gradients to the CPU **separately** (128 GPU→CPU syncs per param, per layer),
  where `SparseGradStore` gathers the active elements.
- **CPU optimizer & streaming.** `SparseAdam` (12 M active params), resident-overlay
  sync, and 4-bit dequant all run on the CPU, serialized, while the GPU sits idle.

Expert compute is a tiny fraction of that, so optimizing it — done three different ways
here — yields nothing.

## Where the speedup actually is

1. **Batch the gradient capture** — one GPU→CPU transfer of the full grad instead of
   128, then vectorize the active-element gather.
2. **Move the optimizer state onto the GPU** so the step doesn't round-trip to CPU.
3. **Cut the GPU↔CPU syncs** on the critical path (dequant, resident sync).

**Caveat — config size.** At seq 256 / microbatch 1 there are only 256 tokens, so fixed
CPU overheads dominate. At a realistic batch (microbatch 4–8, seq 512) GPU compute is a
larger share and the batched GEMM does help — this tiny config (chosen to fit Kaggle's
RAM/time budget) simply can't show it. The batched forward is kept as the right default;
it just isn't what's gating *this* benchmark.

## Reproduce

- [`kaggle_t4_bench.py`](kaggle_t4_bench.py) — `env DISPATCH=batched|loop|dense`
  selects the strategy (`batched` default on CUDA).
- Raw logs: [`sparse_batched_run.log`](sparse_batched_run.log) (batched),
  [`sparse_dispatch_run.log`](sparse_dispatch_run.log) (loop).
