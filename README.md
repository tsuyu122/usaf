# USAF — Ultra Sparse Adaptive Fine-Tuning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-downloads)
[![Status](https://img.shields.io/badge/Status-Beta-orange)]()

**Sparse fine-tuning for Mixture-of-Experts models at 0.5% density.**

Train 24M parameters out of 4.8B — 200x fewer than full fine-tuning — while maintaining
quality through dynamic sparsity (RigL) and router co-training.

Built for hardware-limited researchers. Tested on AMD RX 6750 XT (12GB), NVIDIA T4 (16GB),
and multi-GPU setups. Quantized weights stream from disk in 4-bit — models that would
require 60GB of VRAM fit in 16GB.

---

## Quick Start

```bash
pip install usaf
```

```python
from usaf.train import USAFConfig, train

config = USAFConfig(
    model_path="Qwen/Qwen3-30B-A3B",
    dataset="data/train.jsonl",
    steps=360,
    frac=0.005
)

train(config)
```

Or via CLI:

```bash
usaf train --model Qwen/Qwen3-30B-A3B --dataset data/train.jsonl --steps 360
```

## Features

- **Ultra Sparse** — 0.5% active parameters (24M / 4.8B). 200x fewer than full FT.
- **Adaptive** — RigL dynamic reselection drops underperforming weights and grows new ones.
- **Router Training** — Co-trains the gating network alongside expert weights.
- **Streaming 4-bit** — Expert weights stream from disk in 4-bit, dequantized on-the-fly.
- **Multi-Backend** — CUDA (NVIDIA), DirectML (AMD), CPU fallback. Vulkan acceleration.
- **Multi-GPU** — DataParallel across multiple GPUs with mixed precision (AMP).
- **Any MoE Model** — Auto-detects architecture from HuggingFace config.

## Results

| Benchmark | RX 6750 XT (DML) | T4 (CUDA) | 2x T4 |
|---|---|---|---|
| tok/s | 9 | ~30 | ~50 |
| 180 steps | 7.8h | ~2h | ~1.2h |
| RAM peak | 19GB | 14GB | 14GB/GPU |

- Loss improvement: 1.43 → 1.00 (30% reduction over 180 steps)
- Perplexity (Held-out): 4.52 → 4.24 (6% improvement)
- Zero NaN steps over full training runs

## Supported Backends

| Backend | Status | Notes |
|---|---|---|
| **CUDA** (NVIDIA) | ✓ | AMP, TF32, DataParallel |
| **DirectML** (AMD) | ✓ | Windows only |
| **Vulkan** | ✓ | Cross-vendor, persistent buffers |
| **CPU** | ✓ | Fallback, slow |

## Project Structure

```
usaf/
├── train.py          # Universal trainer with auto-detection
├── moe_loader.py     # Quantized expert streaming + sparse grad capture
├── quantization.py   # 4-bit pack/dequant (HQQ format)
├── sparse_optim.py   # SparseAdam optimizer
├── frozen_cache.py   # Pre-computed hidden states for frozen layers
├── qwen3moe_dml.py   # DirectML patching for Qwen3MoE
├── vk_layer.py       # Vulkan-accelerated layer operations
└── vulkan/           # Vulkan compute shaders + pybind module
```

## Citation

```bibtex
@software{usaf2026,
  title   = {USAF: Ultra Sparse Adaptive Fine-Tuning},
  author  = {Enzo Tsuyoshi},
  year    = {2026},
  license = {Apache-2.0},
  url     = {https://github.com/tsuyu122/usaf}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Contributions require [CLA](CLA.md).

---

*Built by [Enzo Tsuyoshi](https://github.com/tsuyu122)*
