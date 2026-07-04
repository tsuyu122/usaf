# USAF — Ultra Sparse Adaptive Fine-Tuning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org) [![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-downloads) [![Status](https://img.shields.io/badge/Status-Beta-orange)]()

Fine-tune Mixture-of-Experts models on the same GPU you use for inference.

Qwen3-30B-A3B needs ~20GB just to load. Full fine-tuning would need 80GB+. USAF trains 24M out of 4.8B parameters — 200x fewer — using dynamic sparsity (RigL) and router co-training. Tested on AMD RX 6750 XT (12GB) and NVIDIA T4 (16GB).

---

## Results

180 steps on Qwen3-30B-A3B, RX 6750 XT 12GB, DirectML backend.

| Metric | Before | After |
|---|---|---|
| Loss | 1.43 | 1.00 (-30%) |
| In-domain PPL | 2.83 | 2.76 |
| Held-out PPL | 4.52 | 4.24 (-6%) |
| Steps skipped | — | 0 / 180 |

Held-out evaluation uses repositories excluded from training (Flecs, SFML, EnTT, Box2D). Their perplexity improved alongside the training data — the model is generalizing, not memorizing.

## How It Works

The model is split into frozen layers and trainable layers. Frozen layers (0-39 for Qwen3) only run forward passes — their 4-bit weights stream from RAM and are dequantized on the fly. Trainable layers (40-47) keep persistent fp16 copies in RAM to avoid repeated dequantization.

Only 0.5% of expert weights are active at any time. The training loop has four phases:

**Importance.** One dense forward/backward pass computes gradient magnitude for every weight. The top 0.5% per expert tensor are selected as the initial active set.

**Sparse training.** Each step runs a forward pass through all layers, then backward only through trainable layers. Gradients are captured only for active weights. SparseAdam updates the compact master parameters.

**RigL reselection.** Every 50 steps, a dense pass re-scores all weights. Underperforming actives are dropped and replaced by the highest-gradient inactive weights. Optimizer state resets for new weights. Turnover starts at ~92% and decreases as the model converges.

**Router co-training.** The gating network (`mlp.gate.weight`, 2048×128 per layer, 2M total) is trained densely with SGD+momentum. A single step can drop loss by 0.65 (1.80 → 1.15).

## Memory Layout

- **System RAM:** Q4 packed weights for all 48 layers (~3GB). Resident fp16 tensors for 8 trainable layers (~2GB). SparseAdam state (~200MB).
- **GPU VRAM:** Non-expert model weights (~4GB). One active expert layer at a time (~270MB). Attention buffers, gate optimizer state.

Expert weights stay quantized in RAM. Only the currently-executing layer gets dequantized to fp16 on GPU. Trainable layers avoid this via resident mode.

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

Environment variables control everything. No config files needed.

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
| Held-out evaluation | Production |

## Performance

| Hardware | Backend | tok/s | 180 steps |
|---|---|---|---|
| RX 6750 XT 12GB | DirectML | 9 | 7.8h |
| T4 16GB | CUDA | ~30 | ~2h |
| 2× T4 16GB | CUDA | ~50 | ~1.2h |

*CUDA numbers are estimates. Benchmarks on real hardware welcome.*

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers, safetensors, psutil
- GPU with 12GB+ VRAM or 32GB system RAM
- Vulkan SDK (optional, experimental backend)

## Project Structure

```
usaf/
├── train.py              # Universal trainer with auto-configuration
├── moe_loader.py         # Quantized expert streaming, sparse grad capture
├── sparse_optim.py       # SparseAdam optimizer
├── quantization.py       # 4-bit pack/dequant (HQQ format)
├── frozen_cache.py       # Pre-computed hidden states for frozen layers
├── vk_layer.py           # Vulkan persistent buffer integration
├── vulkan/               # Compute shaders and pybind module
└── train_qwen3_12h.py    # Main training script (root)
```

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
