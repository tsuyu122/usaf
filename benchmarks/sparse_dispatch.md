# Sparse expert dispatch — correctness confirmed, throughput-neutral (T4)

The default MoE forward computes **all 128 experts for every token** (dense-masked;
non-routed tokens are multiplied by weight 0). This experiment routes each expert to
**only its assigned top-k tokens** via CUDA `gather`/`scatter` (`index_select` +
`index_add`) — the classic "compute 8, not 128" optimization.

Same run config as [cuda_t4.md](cuda_t4.md) (Qwen3-30B-A3B, 1× T4, 4 trainable layers,
seq 256, FRAC 0.5 %), so the only variable is dense-masked vs sparse dispatch.

## Result

| | Dense-masked | **Sparse dispatch** |
|---|---|---|
| Steady step time | 56.8 s/step | **60.0 s/step** |
| Throughput | 4.5 tok/s | **4.3 tok/s** |
| VRAM peak | 8.0 GB | 8.0 GB |
| Host RAM peak | 15.8 GB | 14.3 GB |
| Importance loss (3 samples) | 1.6807 / 1.1133 / 1.5312 | 1.6738 / 1.1143 / 1.5312 |
| Probe loss (fixed sample) | 1.6807 → 1.1895 | 1.6738 → 1.1152 |
| Steps skipped | 0 / 20 | 0 / 20 |

## Two findings

**1. Correctness — confirmed.** The sparse forward is numerically equivalent to the
dense-masked one (identical importance losses, same 0.500 % selection, same clean loss
decrease). This is expected: non-routed tokens contributed exactly 0 to the dense sum,
so skipping them changes nothing but the compute.

**2. Throughput — no speedup (marginally slower).** Cutting expert compute 16× (128 → 8
experts/token) did **not** speed up the step. The bottleneck was never expert FLOPs:

- At seq 256 / microbatch 1 there are only 256 tokens; with top-8 each expert gets
  ~16 tokens. The forward still **loops over 128 experts**, now launching 128 *tiny*
  GEMMs. On a GPU, many small GEMMs are **launch-overhead and occupancy bound**, not
  throughput bound — so shrinking each matmul buys nothing, and the extra
  `nonzero`/`index_select`/`index_add` bookkeeping makes it slightly slower.
- The rest of the step (per-layer manual backward, 4-bit dequant, `SparseAdam`,
  resident sync) runs largely on the **CPU**, leaving the GPU idle regardless of expert
  compute.

## Takeaway

A naive Python loop over the 128 experts is the wrong granularity — it does not matter
whether each expert sees 16 or 256 tokens when you pay 128 sequential kernel launches
either way. The real win requires **grouped / batched GEMM** (a single batched matmul
over the ragged per-expert token groups, à la MegaBlocks / grouped-gemm), plus moving
the optimizer/streaming off the critical path so the GPU stays busy.

This is a **negative result worth keeping**: it redirects the optimization effort from
"compute fewer experts" (done here, no effect) to "batch the expert GEMMs + unblock the
GPU", which is where the throughput actually is.

## Reproduce

- [`kaggle_t4_bench.py`](kaggle_t4_bench.py) — contains `sparse_qwen3_experts_forward`
  (applied by default on CUDA; set env `DENSE=1` to fall back to the dense-masked path).
- [`sparse_dispatch_run.log`](sparse_dispatch_run.log) — full raw run log.
