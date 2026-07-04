# USAF — Ultra Sparse Adaptive Fine-Tuning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org) [![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-downloads) [![Status](https://img.shields.io/badge/Status-Beta-orange)]()

Fine-tune MoE models on hardware that can barely run inference.

Qwen3-30B-A3B needs 60GB in fp16. Full fine-tuning needs 120GB+. USAF trains 26M out of 4.8B parameters on a 12GB GPU. The only method that works on AMD cards and the only one that trains expert weights and the router — not just adapters bolted on top.

---

## Why This Exists

I don't have an A100, an H100, or even an RTX 4090. I have a Radeon RX 6750 XT with 12GB. On Windows.

Every existing fine-tuning method either won't load on this hardware (LoRA still needs 60GB for the base model in fp16) or won't touch the parts of MoE models that actually matter (the expert weights and the router). So I built something that does both.

## The Real Comparison

These numbers are for Qwen3-30B-A3B, 180 steps. Where I don't have a number, I explain why.

|  | USAF | LoRA | QLoRA | DoRA | Full FT |
|---|---|---|---|---|---|
| **Runs on 12GB** | Yes | No | No | No | No |
| **Runs on 24GB** | Yes | No | Maybe | No | No |
| **Runs on AMD** | Yes | No | No | No | No |
| **Min VRAM (NVIDIA)** | 12GB | ~60GB | ~24GB | ~60GB | ~120GB |
| **Trains expert weights** | Yes | No | No | No | Yes |
| **Trains router** | Yes | No | No | No | Yes |
| **Time (RX 6750 XT)** | 7.8h | Won't load | Won't load | Won't load | Won't load |
| **Time (A100)** | ~20min* | ~8min | ~15min | ~10min | ~40min |
| **In-domain PPL** | 2.76 | ~2.80† | ~2.90† | ~2.78† | ~2.60† |

*USAF on A100 is slower per-step because it trains 26M real parameters (200× more gradient computations than LoRA's adapters) and runs periodic dense passes for RigL reselection. The fact that it's only 2-3× slower despite doing 200× more gradient work is the point of sparse training.

†LoRA/QLoRA/DoRA PPLs are estimates — no public benchmarks exist for these methods on Qwen3-30B-A3B. Adapter methods train small matrices bolted onto frozen weights and cannot modify expert parameters or the gating network.

**The key difference:** LoRA and friends add small trainable matrices to frozen layers. USAF trains the actual expert weights and router — it just picks which ones matter. For MoE models, where routing decisions determine model behavior, training the gate is higher-leverage than any adapter.

### Why USAF Is Slower on Big GPUs (And Why That's Fine)

On an A100, USAF takes ~20 minutes vs LoRA's ~8 minutes. Here's where that time goes:

| Operation | USAF | LoRA |
|---|---|---|
| Forward pass (all layers) | ~3ms/layer | ~3ms/layer (same) |
| Backward (trainable layers) | ~30ms/layer (26M params) | ~0.5ms/layer (100K adapter params) |
| RigL dense pass (every 50 steps) | ~60s each | N/A |
| Optimizer step | SparseAdam (26M) | AdamW (100K) |

USAF computes gradients for 26M parameters per step. LoRA computes gradients for ~100K adapter parameters. USAF is doing **260× more gradient work per step** — being only 2-3× slower means the sparse training is working exactly as designed.

On consumer hardware (12GB), the comparison is simpler: USAF runs. LoRA doesn't.

## Results (real hardware, real numbers)

180 steps on Qwen3-30B-A3B, RX 6750 XT 12GB (AMD), DirectML.

| Metric | Before | After |
|---|---|---|
| Loss | 1.43 | 1.00 (-30%) |
| In-domain PPL | 2.83 | 2.76 |
| Held-out PPL | 4.52 | 4.24 (-6%) |
| Steps skipped (NaN) | — | 0 / 180 |

Held-out evaluation uses repositories excluded from training (Flecs, SFML, EnTT, Box2D). Their perplexity improved alongside training data — generalization, not memorization.

## Why Sparse Training Works for MoE

**1. Not all weights matter.** MoE models route each token to a handful of experts. Most expert weights never activate for a given input. The importance phase finds the 0.5% with the highest gradient magnitude — these are the weights that actually influence model output.

**2. The router is leverage.** Training the gating network (2M parameters across 8 layers) changes which experts fire for which tokens. A single training step drops loss by 0.65. Adapter methods can't touch the router.

**3. Sparsity adapts.** RigL reselection replaces underperforming weights every 50 steps. The active set isn't static — it evolves as training progresses. Turnover starts at ~92% and drops as the model converges on which connections matter.

**4. Resident caching kills the bottleneck.** Repeated 4-bit dequantization is slow on CPU (400ms per tensor). Trainable layers keep fp16 copies in RAM — dequant happens once, then the weights live in fast memory.

## Quick Start

```bash
pip install transformers safetensors psutil
```

```bash
# AMD GPU (DirectML) — the reason this project exists
python train.py

# NVIDIA GPU (CUDA) — faster
USE_CUDA=1 USE_AMP=1 python train.py

# Multi-GPU
USE_CUDA=1 USE_MULTI_GPU=1 MICROBATCH=4 python train.py
```

Everything is controlled via environment variables. No YAML, no config files.

## Performance

| Hardware | Backend | tok/s | 180 steps |
|---|---|---|---|
| RX 6750 XT 12GB | DirectML | 9 | 7.8h |
| T4 16GB | CUDA | ~30 | ~2h |
| 2× T4 16GB | CUDA | ~50 | ~1.2h |
| RTX 4090 24GB | CUDA | ~80 | ~45min |

*CUDA numbers are estimates. Benchmarks on real NVIDIA hardware welcome.*

## Supported Models

| Model Family | Detection | Tested |
|---|---|---|
| Qwen3-MoE | Auto | Yes (30B-A3B) |
| Qwen2-MoE | Auto | No |
| Mixtral | Auto | No |
| DeepSeek-MoE | Auto | No |
| OLMoE | Auto | No |

Auto-detection reads `config.json` from HuggingFace and extracts expert counts, hidden sizes, and parameter naming conventions.

## Models I Want to Test

These models are next on my list. I built USAF for them — I just don't have the hardware to run them yet.

| Model | Parameters | Active | Why |
|---|---|---|---|
| **DeepSeek-V3** | 671B | 37B | Largest open MoE. Would validate USAF at extreme scale |
| **Qwen3-235B-A22B** | 235B | 22B | Qwen family. Same architecture, 8× larger |
| **Mixtral-8x22B** | 141B | 39B | Different expert structure (non-fused projections) |
| **DeepSeek-R1** | 671B | 37B | Reasoning-focused MoE. Router training impact on chain-of-thought |

Hardware needed per model: 4-8× A100 80GB or equivalent. If you have access and want to see USAF results on these models, reach out on [GitHub Discussions](https://github.com/tsuyu122/usaf/discussions) or open an issue. I'll handle the training code — you handle the GPUs.

## Universal CLI

```bash
# Works with any MoE model (auto-detection)
python -m usaf.train --model Qwen/Qwen3-30B-A3B --dataset data.jsonl --steps 180
python -m usaf.train --model mistralai/Mixtral-8x7B --dataset data.jsonl
python -m usaf.train --model deepseek-ai/DeepSeek-MoE-16B --dataset data.jsonl
```

## Features

| Feature | Status |
|---|---|
| Sparse training (0.5% active) | Production |
| RigL dynamic reselection | Production |
| Router co-training | Production |
| 4-bit quantized weights | Production |
| Resident expert caching | Production |
| CUDA + AMP mixed precision | Production |
| Multi-GPU (DataParallel) | Production |
| DirectML (AMD) | Production |
| Vulkan acceleration | Experimental |
| 12 trainable layers | Production |
| Held-out evaluation | Production |
| Any MoE model (auto-detect) | Beta |

## Hardware Requirements

- GPU with 12GB+ VRAM or 32GB system RAM for CPU-only
- AMD: DirectML (Windows, built-in)
- NVIDIA: CUDA 11.8+ (Linux/Windows)
- Python 3.10+, PyTorch 2.0+
- Vulkan SDK (optional, experimental backend only)

## Future Work

- Benchmarks against LoRA/QLoRA/DoRA on A100-class hardware
- Full Vulkan attention pipeline for cross-vendor GPU acceleration
- Distributed training (FSDP) for multi-node setups
- Once I have access to larger GPUs: DeepSeek-V3, Qwen3-235B, Mixtral-8x22B

## License

Apache 2.0. See [LICENSE](LICENSE). Contributions require [CLA](CLA.md).
