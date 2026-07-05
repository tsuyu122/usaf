# USAF — Real CUDA benchmark (NVIDIA Tesla T4)

First **measured** CUDA numbers for USAF (the throughput figures in the main
README were estimates). Run on Kaggle (2× T4 allocation, single T4 used — see
caveats). Reproduces end-to-end on stock NVIDIA + CUDA after the grad-capture
fix in [#1](https://github.com/tsuyu122/usaf/pull/1).

## Setup

| | |
|---|---|
| Model | Qwen3-30B-A3B (30B total / 3B active, 128 experts, 48 layers) |
| GPU | 1× NVIDIA Tesla T4 (16 GB, sm_75) |
| Backend | CUDA + AMP + TF32, PyTorch (Kaggle CUDA 12.x), transformers 5.12.1 |
| Trainable | 4 layers (44–47), FRAC = 0.5 % → 12,079,588 active params |
| Seq len | 256 |
| Host RAM | 32 GB cgroup limit |
| Optimizations | frozen-activation cache + `free_frozen` (resident q4 dropped after precompute) |

## Results

| Metric | Value |
|---|---|
| **Throughput (steady state)** | **4.5 tok/s** (56.8 s/step) |
| **VRAM peak** | **8.0 GB** |
| **Host RAM peak** | 15.8 GB |
| Sparse selection | 12,079,588 / 2,415,919,104 = **0.500 %** |
| **Probe loss (fixed sample, no_grad)** | **1.6807 → 1.1895 (−0.49)** |
| Steps skipped (NaN/inf) | **0 / 20** |
| Frozen precompute | 142 s/sample |
| 180-step extrapolation | ~2.8 h |

## What this proves

- **USAF trains on stock NVIDIA/CUDA.** Every optimizer step applied (0 skipped),
  sparse gradients were captured, and the loss on a held-out fixed sample dropped
  1.68 → 1.19 — the weights actually learned.
- **8 GB VRAM footprint** → runs on any ≥ 12 GB NVIDIA card. (Fits a T4 with room
  to spare.)
- The exact 0.500 % selection and the grad-capture path work identically to the
  DirectML path.

## Honest caveats

- **Single T4.** The training loop runs decoder layers manually (per-layer
  forward/backward), which bypasses `DataParallel`, so this measures one T4. A
  real 2× T4 speedup needs model/pipeline parallelism (future work).
- **This corrects the README estimate of "~30 tok/s" on T4** — real throughput is
  single-digit, because the current MoE forward is *dense-masked* (it computes all
  128 experts for every token, ~16× the necessary compute). Sparse expert dispatch
  is the biggest remaining speedup (see the optimization roadmap).
- **Benchmark config, not a full run.** 4 trainable layers, seq 256, a tiny
  4-sample pool overfit for 5 epochs to demonstrate learning quickly. The full
  DirectML 12 h run in the main README remains the at-scale quality result.
- **Host RAM is the binding constraint on Kaggle.** The 15 GB q4 expert dict plus
  dequant peaks brush the 32 GB kernel cgroup limit; the frozen-activation cache +
  `free_frozen` were required to fit. A normal NVIDIA workstation (24 GB GPU +
  32 GB RAM) runs the standard streaming path without this.

## Reproduce

- [`kaggle_t4_bench.py`](kaggle_t4_bench.py) — the exact benchmark script (Kaggle
  kernel; auto-resolves dataset mounts, pins transformers, applies the
  grad-capture patch, precomputes frozen activations, then trains the trainable
  layers).
- [`cuda_t4_run.log`](cuda_t4_run.log) — full raw run log.

Push with `kaggle kernels push -p . --accelerator NvidiaTeslaT4` (the `NvidiaTeslaT4`
accelerator is Kaggle's 2× T4 config; the default P100 is sm_60 and rejected by
current PyTorch).
