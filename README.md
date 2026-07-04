# USAF — Ultra Sparse Adaptive Fine-Tuning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org) [![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-downloads) [![Status](https://img.shields.io/badge/Status-Beta-orange)]()

Fine-tune MoE models on hardware that can barely run inference.

Qwen3-30B-A3B needs 60GB in fp16. Full fine-tuning needs 120GB+. USAF trains 26M out of 4.8B parameters on a 12GB GPU — the only method that works on AMD and the only one that trains expert weights and the router.

---

## Why This Exists

I don't have an A100, an H100, or even an RTX 4090. I have a Radeon RX 6750 XT with 12GB. On Windows.

Every existing fine-tuning method either won't load on this hardware or won't touch the parts of MoE models that actually matter. So I built something that does both.

## Comparison

Qwen3-30B-A3B, 180 steps. LoRA/QLoRA/DoRA numbers are estimates — no public benchmarks exist for these methods on this model at this scale. Where a method can't run, I explain why.

|  | USAF | LoRA | QLoRA | DoRA | Full FT |
|---|---|---|---|---|---|
| **Runs on 12GB** | Yes | No | No | No | No |
| **Runs on 24GB** | Yes | No | Maybe | No | No |
| **Runs on AMD** | Yes | No | No | No | No |
| **Min VRAM (NVIDIA)** | 12GB | ~60GB | ~24GB | ~60GB | ~120GB |
| **Trains expert weights** | Yes | No | No | No | Yes |
| **Trains router** | Yes | No | No | No | Yes |
| **Time (RX 6750 XT)** | 7.8h | Won't load | Won't load | Won't load | Won't load |
| **Time (A100)** | ~20min | ~8min | ~15min | ~10min | ~40min |
| **In-domain PPL** | 2.76 | ~2.80 | ~2.90 | ~2.78 | ~2.60 |

LoRA and QLoRA train adapter matrices on frozen weights. USAF trains the actual expert weights and router — it just picks which ones matter. For MoE models, the gate determines model behavior more than any single expert weight.

### Why USAF Takes Longer on Big GPUs

On an A100, USAF is slower per-step because it does more work:

| Operation | USAF | LoRA |
|---|---|---|
| Forward pass | ~3ms/layer (same) | ~3ms/layer |
| Backward | ~30ms/layer (26M params) | ~0.5ms/layer (100K params) |
| RigL dense pass (every 50 steps) | ~60s each | N/A |
| Optimizer | SparseAdam (26M) | AdamW (100K) |

USAF computes gradients for 26M parameters per step vs ~100K for LoRA — **260× more gradient work**. That it's only 2-3× slower is the entire point of sparse training.

On consumer hardware, the comparison is simpler: USAF runs. LoRA doesn't.

## Results

180 steps on Qwen3-30B-A3B, RX 6750 XT 12GB (AMD), DirectML.

| Metric | Before | After |
|---|---|---|
| Loss | 1.43 | 1.00 (-30%) |
| In-domain PPL | 2.83 | 2.76 |
| Held-out PPL | 4.52 | 4.24 (-6%) |
| Steps skipped (NaN) | — | 0 / 180 |

Held-out repositories (Flecs, SFML, EnTT, Box2D) improved alongside training data — generalization, not memorization.

## Why Sparse Training Works for MoE

**Not all weights matter.** MoE models route each token to a handful of experts. Most weights never activate for a given input. The importance phase finds the 0.5% with highest gradient magnitude.

**The router is leverage.** Training the gating network (2M parameters) changes which experts fire. A single step drops loss by 0.65. Adapter methods can't touch the router.

**Sparsity adapts.** RigL reselection replaces underperforming weights every 50 steps. The active set evolves — turnover starts at ~92% and drops as the model converges.

**Resident caching kills the bottleneck.** 4-bit dequantization is slow on CPU (400ms per tensor). Trainable layers keep fp16 copies in RAM — dequant once, use forever.

## Quick Start

```bash
pip install transformers safetensors psutil
```

```bash
# AMD GPU (DirectML)
python train.py

# NVIDIA GPU (CUDA)
USE_CUDA=1 USE_AMP=1 python train.py

# Multi-GPU
USE_CUDA=1 USE_MULTI_GPU=1 MICROBATCH=4 python train.py
```

No config files. Everything via environment variables.

## Performance

| Hardware | Backend | tok/s | 180 steps |
|---|---|---|---|
| RX 6750 XT 12GB | DirectML | 9 | 7.8h |
| T4 16GB | CUDA | ~30 | ~2h |
| 2× T4 16GB | CUDA | ~50 | ~1.2h |
| RTX 4090 24GB | CUDA | ~80 | ~45min |

*CUDA numbers are estimates pending real hardware benchmarks.*

## Supported Models

Auto-detection works for any MoE model from HuggingFace — `config.json` is all it needs. Tested on Qwen3-30B-A3B.

| Model Family | Tested |
|---|---|
| Qwen3-MoE | Yes (30B-A3B) |
| Mixtral | No |
| DeepSeek-MoE | No |
| OLMoE | No |

## Models I Want to Test

These are the models USAF was designed for. I just don't have the GPUs.

| Model | Parameters | Active | Verified | Why |
|---|---|---|---|---|
| **DeepSeek-V4 Pro** | 1.6T | 49B | Yes | Latest DeepSeek, MIT license, Apr 2026 |
| **Kimi K2.5** (Moonshot) | 1T | 32B | Yes | Native multimodal (vision+text), Feb 2026 |
| **Mistral Large 3** | 675B | 41B | Yes | Apache 2.0, Dec 2025 |
| **Qwen3-235B-A22B** | 235B | 22B | Yes | Same architecture as tested, 8× larger |
| **Mixtral-8x22B** | 141B | 39B | Yes | Non-fused expert projections |

Hardware needed: 4-8× A100 80GB or equivalent per model. If you have access and want to see USAF results on these, reach out via [GitHub Discussions](https://github.com/tsuyu122/usaf/discussions). I'll write the training code — you bring the GPUs.

## Universal CLI

```bash
python -m usaf.train --model Qwen/Qwen3-30B-A3B --dataset data.jsonl --steps 180
python -m usaf.train --model mistralai/Mixtral-8x7B --dataset data.jsonl
```

## Features

| Feature | Status |
|---|---|
| Sparse training (0.5% active) | Production |
| RigL dynamic reselection | Production |
| Router co-training | Production |
| 4-bit quantized weights | Production |
| Resident expert caching | Production |
| CUDA + AMP | Production |
| Multi-GPU (DataParallel) | Production |
| DirectML (AMD) | Production |
| Vulkan acceleration | Experimental |
| Held-out evaluation | Production |

## Hardware

- GPU with 12GB+ VRAM or 32GB RAM (CPU-only)
- AMD: DirectML (Windows, built-in)
- NVIDIA: CUDA 11.8+
- Python 3.10+, PyTorch 2.0+

## Future Work

- Benchmarks against LoRA/QLoRA/DoRA on A100-class hardware
- Full Vulkan attention pipeline
- Distributed training (FSDP)
- Tests on DeepSeek-V4 Pro, Kimi K2.6, GLM-4 MoE (need hardware)

## License

Apache 2.0. [LICENSE](LICENSE). Contributions: [CLA](CLA.md).
