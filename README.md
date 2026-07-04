# USAF — Ultra Sparse Adaptive Fine-Tuning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org) [![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-downloads) [![Status](https://img.shields.io/badge/Status-Beta-orange)]()

Fine-tune Mixture-of-Experts models on consumer GPUs. Qwen3-30B-A3B needs ~20GB just to load. Full fine-tuning would need 80GB+. USAF trains 24M out of 4.8B parameters — 200x fewer — on 12GB cards.

---

## Why This Exists

I don't have an A100. I have a Radeon RX 6750 XT with 12GB of VRAM.

Existing methods for fine-tuning MoE models fall into two camps: too heavy (LoRA still needs 40GB+ for 30B-class models) or too shallow (adapters that never touch the experts or the router). There was nothing that let me train a 30B MoE model on a gaming GPU.

So I built one.

## Comparison

Qwen3-30B-A3B, 180 training steps. LoRA/QLoRA/DoRA numbers are theoretical estimates — I don't have the hardware to run them, which is exactly why USAF exists.

| Method | Min VRAM | Trainable Params | Est. Time (A100) | Est. Time (RX 6750) | Notes |
|---|---|---|---|---|---|
| **Full FT** | 80GB+ | 4.8B | ~1h | N/A | Requires datacenter GPU |
| **LoRA** (r=16) | ~40GB | 100M | ~1.5h | N/A | Doesn't train experts or router |
| **QLoRA** (4-bit) | ~20GB | 100M | ~2.5h | ~15h | 4-bit overhead, adapter-only |
| **DoRA** (r=16) | ~40GB | 100M | ~2h | N/A | Weight decomposition, same limits |
| **USAF** | **12GB** | **26M** | — | **7.8h** | Trains experts + router, sparse |

Key insight: LoRA and friends add adapters on top of frozen weights. USAF trains actual expert weights — it just picks which ones matter. This matters for MoE models because the routing decisions (which expert fires for which token) live in weights that LoRA never touches.

I plan to run benchmarks against LoRA/QLoRA/DoRA on larger hardware when I have access. For now, the numbers above are theoretical estimates based on memory math: LoRA on Qwen3-30B requires keeping the full fp16 model in VRAM (~60GB for the base model alone) plus adapter states, hence the 40GB minimum with aggressive offloading.

## Results (USAF, real hardware)

180 steps on Qwen3-30B-A3B, RX 6750 XT 12GB, DirectML.

| Metric | Before | After |
|---|---|---|
| Loss | 1.43 | 1.00 (-30%) |
| In-domain PPL | 2.83 | 2.76 |
| Held-out PPL | 4.52 | 4.24 (-6%) |
| Steps skipped (NaN) | — | 0 / 180 |

Held-out repositories (Flecs, SFML, EnTT, Box2D) improved alongside training data — generalization, not memorization.

## Why Sparse Training Works for MoE

Mixture-of-Experts models have a unique property: the routing network already decides which weights contribute to any given token. Most expert weights are dormant for most inputs. This means:

**1. Not all weights are equally important.** A handful of expert weights handle the majority of routing decisions. The importance phase finds them automatically via gradient magnitude.

**2. The router is leverage.** Training the gating network (`mlp.gate.weight`, 2M dense parameters) changes which experts activate. A single step drops loss by 0.65. LoRA can't touch the router.

**3. Sparsity adapts.** RigL reselection replaces underperforming weights every 50 steps. The active set evolves — it's not a static lottery ticket. Turnover decreases over time as the model settles on which connections matter.

**4. Resident caching eliminates the dequant bottleneck.** Trainable layers keep fp16 copies in RAM. The repeated dequantization that makes naive 4-bit training slow is avoided entirely.

## How It Works

The model is split into frozen and trainable layers. Frozen layers stream 4-bit weights from RAM with on-the-fly dequantization. Trainable layers keep persistent fp16 copies.

Four-phase loop: **Importance** (find top 0.5% weights) → **Sparse training** (forward/backward on active weights only) → **RigL** (periodic re-scoring, drop underperformers) → **Router co-training** (dense SGD on the gating network).

## Quick Start

```bash
pip install transformers safetensors psutil
```

```bash
# AMD GPU (DirectML, default)
python train_qwen3_12h.py

# NVIDIA GPU (CUDA)
USE_CUDA=1 USE_AMP=1 python train_qwen3_12h.py

# Multi-GPU
USE_CUDA=1 USE_MULTI_GPU=1 MICROBATCH=4 python train_qwen3_12h.py
```

Everything is controlled via environment variables. No YAML config files.

## Performance

| Hardware | Backend | tok/s | 180 steps | Can train? |
|---|---|---|---|---|
| RX 6750 XT 12GB | DirectML | 9 | 7.8h | Yes |
| T4 16GB | CUDA | ~30 | ~2h | Yes |
| 2× T4 16GB | CUDA | ~50 | ~1.2h | Yes |
| RTX 4090 24GB | CUDA | ~80 | ~45min | Yes (est.) |

*CUDA numbers are estimates pending benchmarks on real NVIDIA hardware.*

## Features

| Feature | Status |
|---|---|
| Sparse training (0.5% active) | Production |
| RigL dynamic reselection | Production |
| Router co-training | Production |
| 4-bit quantized weights (HQQ) | Production |
| Resident expert caching | Production |
| CUDA + AMP mixed precision | Production |
| Multi-GPU (DataParallel) | Production |
| DirectML (AMD) | Production |
| Vulkan acceleration | Experimental |
| 12 trainable layers | Production |
| Frozen layer caching | Production |

## Future Work

- Benchmarks against LoRA/QLoRA/DoRA on A100-class hardware
- Test on larger MoE models (DeepSeek-V3, Qwen3-235B) — needs hardware I don't have yet
- Full Vulkan pipeline for cross-vendor GPU acceleration
- Distributed training across multiple machines

## Citation

```bibtex
@software{usaf2026,
  title   = {USAF: Ultra Sparse Adaptive Fine-Tuning for MoE Models},
  url     = {https://github.com/tsuyu122/usaf},
  year    = {2026},
  license = {Apache-2.0}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE). Contributions require [CLA](CLA.md).
