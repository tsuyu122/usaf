# USAF — Ultra Sparse Adaptive Fine-Tuning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org) [![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-downloads) [![Status](https://img.shields.io/badge/Status-Beta-orange)]()

**Fine-tune Mixture-of-Experts models on the same GPU you use for inference.**

Qwen3-30B-A3B needs ~20GB just to load. Full fine-tuning would need 80GB+. USAF trains 24M out of 4.8B parameters — **200x fewer** — using dynamic sparsity (RigL) and router co-training. Tested on AMD RX 6750 XT (12GB) and NVIDIA T4 (16GB).

---

## The Problem

MoE models are huge but most of their knowledge lives in the routing patterns, not every expert weight.
Full fine-tuning is prohibitively expensive. LoRA helps but doesn't touch the experts or the router.

**USAF's approach:** pick the 0.5% of weights that matter most, train them dynamically,
and let RigL discover new important connections as training progresses.

## Results (180 steps, Qwen3-30B-A3B, RX 6750 XT 12GB)

| Metric | Before | After |
|---|---|---|
| Loss | 1.43 | **1.00** (-30%) |
| In-domain PPL | 2.83 | **2.76** |
| Held-out PPL | 4.52 | **4.24** (-6%) |
| Steps skipped (NaN) | — | **0 / 180** |

Held-out repos (Flecs, SFML, EnTT, Box2D) improved alongside training data — the model is **generalizing**, not memorizing.

## Quick Start

```bash
pip install transformers safetensors psutil
```

```bash
# AMD GPU (DirectML) — default
python train_qwen3_12h.py

# NVIDIA GPU (CUDA) — faster
USE_CUDA=1 USE_AMP=1 python train_qwen3_12h.py

# 2x GPU (DataParallel)
USE_CUDA=1 USE_MULTI_GPU=1 MICROBATCH=4 python train_qwen3_12h.py
```

## How It Works

```
┌─ Frozen Layers (0-39) ──────────────────────────────────┐
│  Streaming 4-bit dequant. Weights live on disk.        │
│  Forward pass only — no gradients stored.              │
│  42% of step time. Optional frozen cache available.    │
└─────────────────────────────────────────────────────────┘
┌─ Trainable Layers (40-47) ─────────────────────────────┐
│  Resident fp16 tensors in RAM. SparseAdam on 24M       │
│  active parameters. RigL reselection every 50 steps.   │
│  Router gates trained densely with SGD+momentum.      │
└─────────────────────────────────────────────────────────┘
```

1. **Importance Phase** — one dense forward/backward to find high-gradient weights.
2. **Active Selection** — keep top 0.5% per expert tensor.
3. **Training Loop** — sparse forward+backward on active weights only.
4. **RigL** — periodically re-score all weights, drop underperformers, grow new ones.
5. **Router Co-Training** — the gating network learns alongside expert weights.

## Technical Details

### Memory Model

```
System RAM (32GB):                     GPU VRAM (12GB):
┌──────────────────────────┐           ┌──────────────────────┐
│ Q4 dict (48 layers)      │           │ Non-expert weights    │
│ Resident fp16 (8 layers) │──PCIe──→  │ Attention buffers     │
│ SparseAdam state         │           │ Current expert (268MB)│
│ SparseGradStore.compact  │           │ Gate optimizer state  │
│ Gate masters (CPU)       │           │ RoPE cos/sin          │
└──────────────────────────┘           └──────────────────────┘
```

Expert weights stay quantized in RAM. Only the currently-executing layer is dequantized to fp16 on GPU. After the forward pass, it's evicted. Trainable layers keep a persistent fp16 copy in RAM (resident mode) to avoid repeated dequantization.

### Sparsity Math

```
Total expert parameters:     4,831,838,208
Active (0.5%):                  24,159,176
Router gates (dense):            2,097,152
──────────────────────────────────────────
Total trained:                  26,256,328  (0.54% of expert params)
```

Each expert tensor (gate_up_proj, down_proj) has 128 experts fused. The top 0.5% of elements across ALL experts are selected — not per-expert. This means some experts get more active weights than others, naturally allocating capacity where gradients are strongest.

### RigL Reselection

Every 50 steps, a dense forward/backward pass re-scores all weights. The bottom ~50% of active weights are dropped, and the top inactive weights take their place. Optimizer state (momentum, variance) is reset for new weights. Turnover decreases over time (92% → 82% in 180 steps) as the model converges on which weights matter.

### Router Training

The `mlp.gate.weight` (2048×128 per layer) is trained densely with SGD+momentum alongside the sparse expert weights. This lets the router adapt which experts are activated for each token. A single step can drop the loss by 0.65 (from 1.80 to 1.15), showing the router has significant leverage over model behavior.

## Features

| Feature | Status |
|---|---|
| Sparse training (0.5% active) | ✓ |
| RigL dynamic reselection | ✓ |
| Router co-training | ✓ |
| 4-bit quantized weights (HQQ) | ✓ |
| Resident expert caching | ✓ |
| CUDA + AMP mixed precision | ✓ |
| Multi-GPU (DataParallel) | ✓ |
| DirectML (AMD) | ✓ |
| Vulkan acceleration | ⚠ Experimental |
| 12 trainable layers | ✓ |
| Frozen cache (eval) | ✓ |
| Held-out evaluation | ✓ |

## Backends

| Backend | Quality | Notes |
|---|---|---|
| **CUDA** | Production | AMP, TF32, DataParallel |
| **DirectML** | Production | Windows, AMD GPUs |
| **Vulkan** | Experimental | Buffer API + dequant work. Attention kernel incomplete. |
| **CPU** | Fallback | Functional but slow |

## Project Structure

```
usaf/
├── train.py              # Universal trainer with auto-configuration
├── moe_loader.py         # Quantized expert streaming + sparse grad capture
├── sparse_optim.py       # SparseAdam optimizer
├── quantization.py       # 4-bit pack/dequant (HQQ format)
├── frozen_cache.py       # Pre-computed hidden states
├── vk_layer.py           # Vulkan persistent buffer integration
└── vulkan/               # Compute shaders + pybind module
```

## Performance Benchmarks

| Hardware | Backend | tok/s | 180 steps |
|---|---|---|---|
| RX 6750 XT 12GB | DirectML | 9 | 7.8h |
| T4 16GB | CUDA | ~30 | ~2h |
| 2x T4 16GB | CUDA | ~50 | ~1.2h |

*CUDA numbers estimated. Benchmarks welcome via PR.*

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers, safetensors, psutil
- GPU with 12GB+ VRAM (or 32GB system RAM for CPU-only)
- Vulkan SDK (optional, for experimental acceleration)

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

Apache 2.0 — see [LICENSE](LICENSE). Contributions require [CLA](CLA.md).
