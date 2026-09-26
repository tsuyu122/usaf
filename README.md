# USAF — Ultra Sparse Adaptive Fine-Tuning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org) [![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-downloads) [![Status](https://img.shields.io/badge/Status-Alpha_1.0-orange)]()

Fine-tune MoE models on hardware that can barely run inference.

Qwen3-30B-A3B (used in all benchmarks on this page) needs 60GB in fp16. Full fine-tuning needs 120GB+. USAF trains 24,159,176 out of 4,831,838,208 expert parameters (0.5%) on a 12GB GPU, plus the router — the only method here that works on AMD and that trains expert weights and the router.

---

## Why This Exists

I don't have an A100, an H100, or even an RTX 4090. I have a Radeon RX 6750 XT with 12GB. On Windows.

Every existing fine-tuning method either won't load on this hardware or won't touch the parts of MoE models that actually matter. So I built something that does both.

## Comparison

Qwen3-30B-A3B, 180 steps. **Only the USAF column was measured.** Every other
entry is a reasoned expectation, marked with `~`, not a benchmark: no
side-by-side run of these methods on this model at this scale exists, and
this project did not run one.

|  | USAF | LoRA | QLoRA | DoRA | Full FT |
|---|---|---|---|---|---|
| **Runs on 12GB** | measured | Won't load | Won't load | Won't load | Won't load |
| **Runs on 24GB** | expected | Won't load | ~ (fits in 4-bit) | Won't load | Won't load |
| **Runs on AMD** | measured | No | No | No | No |
| **Min VRAM (NVIDIA)** | measured (12GB) | ~60GB | ~24GB | ~60GB | ~120GB |
| **Trains expert weights** | Yes | No | No | No | Yes |
| **Trains router** | Yes | No | No | No | Yes |
| **Time (RX 6750 XT)** | measured 7.64h | Won't load | Won't load | Won't load | Won't load |
| **Time (A100)** | not measured | not measured | not measured | not measured | not measured |
| **In-domain PPL** | measured 2.76 | ~2.80 | ~2.90 | ~2.78 | ~2.60 |

LoRA and QLoRA train adapter matrices on frozen weights. USAF trains the actual expert weights and router — it just picks which ones matter. For MoE models, the gate determines model behavior more than any single expert weight.

### Why USAF Takes Longer on Big GPUs

USAF does more work per step than an adapter method, because it computes
gradients for a fixed fraction of every expert tensor rather than for a pair of
small adapters:

| Operation | USAF | LoRA |
|---|---|---|
| Active parameters | 24,159,176 (0.5% of experts) | ~100K (adapters only) |
| Optimizer | SparseAdam over the active set | AdamW over the adapters |
| RigL dense pass (every 50 steps) | 214-219s (measured, 3 passes in this run) | N/A |

That is roughly **240x more trainable parameters** than a typical LoRA
configuration, which is the trade this method makes: quality for throughput.
The per-layer timings below were never measured on an A100 and are therefore
not quoted here.

## Results

180 steps on Qwen3-30B-A3B, RX 6750 XT 12GB (AMD), DirectML.

| Metric | Before | After |
|---|---|---|
| Loss (first step) | 1.4316 | 0.9985 (final step) |
| Loss (mean of last 10 steps) | — | 1.1654 (-18.6% vs first step) |
| In-domain PPL | 2.83 | 2.76 |
| Held-out PPL | 4.52 | 4.24 (-6.2%) |
| Steps skipped (NaN) | — | 0 / 180 |
| Wall clock | — | 7.64h (117s/step median) |
| Peak RAM | — | 19.0GB |

The headline loss move depends on what you average. The final step lands at
0.9985, but single steps are noisy (the run's minimum is 0.5742 and step 2
was *higher* than step 1), so quoting the last step alone overstates the
improvement. Against the mean of the last 10 steps the reduction is 18.6%,
which is the number to trust.

The held-out set is a separate split drawn from repositories outside the
training distribution, so its improvement is evidence of generalization
rather than memorization. The per-repository breakdown is written to
`results/qwen3_12h.json` as `heldout_repos`.

## Why Sparse Training Works for MoE

**Not all weights matter.** MoE models route each token to a handful of experts. Most weights never activate for a given input. The importance phase finds the 0.5% with highest gradient magnitude.

**The router is leverage.** Training the gating network changes which experts fire, and adapter methods cannot reach it. The router is trained alongside the sparse experts with its own SGD optimizer.

**Sparsity adapts.** RigL reselection replaces underperforming weights every 50 steps. The active set stays fixed at 0.5% of experts, but its membership churns: across the three reselections in the reference run, 1.9M, 1.4M and 4.3M of the 24.16M active elements were retained, i.e. turnover of roughly 82-94% per pass.

**Resident caching kills the bottleneck.** 4-bit dequantization is slow on CPU (400ms per tensor). Trainable layers keep fp16 copies in RAM, so a layer is dequantized once into resident storage and every later forward reads the fp16 copy. Residency is on by default (`USE_RESIDENT=1`); when more than 8 layers are trainable only `gate_up_proj` is kept resident, because keeping every expert tensor in fp16 would exceed RAM.

## Quick Start

```bash
# Installs torch, transformers, safetensors, psutil and the package itself.
pip install -e .

# For AMD/Windows GPUs, additionally:
pip install -e ".[dml]"
```

```bash
# AMD GPU (DirectML) - the configuration the reference run used
python train.py

# NVIDIA GPU (CUDA) - unbenchmarked on this machine
USE_CUDA=1 USE_AMP=1 python train.py

# Multi-GPU (CUDA)
USE_CUDA=1 USE_MULTI_GPU=1 python train.py
```

No config files. Everything via environment variables — see the reference
table below for the full list and the real defaults.

## Performance

| Hardware | Backend | tok/s | 180 steps |
|---|---|---|---|
| RX 6750 XT 12GB | DirectML | 8.5 | 7.64h |
| NVIDIA (any) | CUDA | not measured | not measured |

The RX 6750 XT row is the reference run, logged in `logs/qwen3_12h_final.jsonl`.
No CUDA run has been performed on this machine, so no CUDA throughput is
quoted — the code path exists but is unbenchmarked. The 117s/step median is
dominated by the 6 RigL reselections (1140-1220s each, ~1h of the 7.64h total);
ordinary steps run at 107-122s.

## Supported Models

The expert layout is detected automatically. From transformers 4.53 onward every
supported family exposes the same fused expert container, so the naming
convention is no longer hardcoded:

| Model Family | Status |
|---|---|
| Qwen3-MoE | Full run on Qwen3-30B-A3B (RX 6750 XT) |
| Mixtral | Verified end-to-end on a small synthetic Mixtral |
| OLMoE | Verified by the test suite on a synthetic 16-layer OLMoE |
| DeepSeek-MoE | Same fused layout; not run on real weights |

"Verified end-to-end" means the full pipeline — importance selection, sparse
gradients, optimizer step, router update — runs and selects a nonzero active
set. It does not mean the reference numbers below were reproduced on that
family; only Qwen3-30B-A3B was run at real scale.

## Models I Want to Test

These are the models USAF was designed for. I just don't have the GPUs.

| Model | Parameters | Active | Verified | Why |
|---|---|---|---|---|
| **DeepSeek-V4 Pro** | 1.6T | 49B | No | Latest DeepSeek |
| **Kimi K2.5** (Moonshot) | 1T | 32B | No | Native multimodal (vision+text) |
| **Mistral Large 3** | 675B | 41B | No | Apache 2.0 |
| **Qwen3-235B-A22B** | 235B | 22B | No | Same architecture as tested, 8x larger |
| **Mixtral-8x22B** | 141B | 39B | No | Larger Mixtral |

**None of these have been run.** They are targets, not results. None has been
trained, benchmarked, or even downloaded, and the "Verified" column is
uniformly No by design so this table cannot be misread as a benchmark.

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
| CUDA | Code complete, unbenchmarked (no NVIDIA hardware on the reference machine) |
| Multi-GPU (DataParallel) | Code complete, unbenchmarked |
| DirectML (AMD) | Production - the configuration behind every number on this page |
| fp16 + manual loss scaling | Production (used on both the DirectML and CUDA paths) |
| Vulkan kernels (dequant, GEMM, RMSNorm, RoPE, attention) | Built and numerically tested against PyTorch |
| Held-out evaluation | Production |

## What is actually verified

The table above distinguishes code-complete from measured. Here is what the
test suite and the end-to-end runs on this machine cover, so you know exactly
what the numbers rest on.

**Verified numerically, against an independent reference:**

- The sparse gradient is identical to dense autograd. A three-layer stack is
  differentiated twice over the same graph, once with ordinary autograd and
  once with no graph retained, capturing only the active slice per expert via
  the same per-expert tensor hooks the real forward uses. Every active position
  matches. This is the claim the whole method rests on.
- All five Vulkan kernels (attention, RMSNorm, GEMM, RoPE, 4-bit dequant)
  against PyTorch, on a real AMD Radeon RX 6750 XT.
- The composed Vulkan Q/K/V projection (`VKLayer.forward_qkv`), which is the
  path the root `train.py` actually runs, against the PyTorch projections it
  replaces, including the flattened `[B*S, out]` layout its caller reshapes.
  Getting a transpose or a buffer size wrong there would only have shown up
  as a model that trained badly.

  One thing this does *not* cover: `usaf/qwen3_layer_vk.py` composes those
  kernels into a whole decoder layer, and no entry point calls it. It is a
  library, exercised by its own tests, not part of a training run. The Vulkan
  path that a run takes is the Q/K/V projection in `usaf/vk_layer.py`.
- The 4-bit quantizer round-trips across five tensor shapes, packs exactly two
  values per byte, and reaches 3.76x compression.
- RoPE must be a rotation: the test asserts the per-head norm is preserved, not
  just that some number came out.

**Verified end to end on small fixtures** (a 4-layer Qwen3-MoE, a 4-layer
Mixtral and a 4-layer Qwen3 with head_dim 8):

- Training, and the loss going down.
- All 31 arguments of the universal CLI, individually and in combination.
  Four of them were accepted and then ignored, and now do what they say:
  `--frozen-cache`, `--eval-every` (periodic perplexity on the evaluation split),
  `--log-dir` (one JSON object per step, appended, so a long run can be
  inspected while it is still going), and `--eval-report`, which previously only
  worked together with `--eval-only` and so could not report on the weights a
  training run had just produced.
- `--resume`, `--export`, `--eval-only`, `--eval-report`, `--save-every`,
  `--checkpoint-dir`, reselection and accumulation. A resumed run restores the
  active set, the trained weights, the Adam moments and the step counter, and
  resumes at the right point in the LR schedule. It does not checkpoint the
  sampler, so a resumed run sees a different sample order than an uninterrupted
  one: same steps, same optimizer, different data, so the losses will not match
  a continuous run step for step.
- The frozen cache: the cached hidden state is compared against a fresh forward
  of the same prefix and is bit-identical, a run with the cache and a run
  without it reach the same loss, and the cache is reused across processes
  rather than rebuilt.
- The quantized export carries the trained values, and leaves every position
  outside the active set bit-identical **except in the quantization group that
  holds it**. A group is 128 weights sharing one scale and zero point, so a
  group containing a trained value has to re-derive that scale, and the other
  ~127 weights in it move with it. That is inherent to per-group 4-bit
  quantization, not something the exporter can work around. What is guaranteed:
  the 4-bit codes and the scale/zero of every group with no trained weight in it
  are restored byte-for-byte, so an untouched group is bit-identical to the
  original. Measured on the 4-layer fixture, 0 of 4256 untouched groups changed
  (this was 46777 drifted weights before the exporter learned to leave them
  alone).
- Every user-facing error path, each of which previously failed somewhere
  unrelated to the actual mistake. `--cuda` on a machine without an NVIDIA
  GPU now exits 1 with what torch saw and what to do instead, and does so
  before announcing a backend, rather than printing `Backend: CUDA` and then
  dying on a bare `assert` that `python -O` would have removed entirely.

**Not verified here:**

- The CUDA and multi-GPU paths have never been executed - the reference
  machine has no NVIDIA GPU. They are code-complete and unbenchmarked.
- The DirectML path is the configuration behind the numbers on this page, but
  the `torch-directml` build installed on the reference machine fails to load
  against torch 2.13, so the runs below fell back to CPU. The numbers come from
  the DirectML runs recorded before that.
- The 30B-scale results are from the author's runs, not reproducible from this
  repository on a laptop.

## Hardware

- GPU with 12GB+ VRAM or 32GB RAM (CPU-only)
- AMD: DirectML (Windows, built-in)
- NVIDIA: CUDA 11.8+
- Python 3.10+, PyTorch 2.0+

### Optional: building the Vulkan kernels

`USE_VK=1` needs the `usaf_vk` extension. It is not required; without it the
run uses the PyTorch path. To build it you need the Vulkan SDK (for `glslc`)
and CMake 3.20+:

```bash
pip install pybind11
cmake -S usaf/vulkan -B build -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE=$(which python)
cmake --build build --config Release
PYTHONPATH=build/Release python -m pytest tests/test_vulkan_kernels.py -q
```

Pass `-DPython3_EXECUTABLE` explicitly. If you do not, CMake picks whatever
interpreter is newest on the machine, which on a typical Windows box is the
Store build of a newer Python that has no `pybind11`, and the configure step
fails. The build stages the compiled shaders in `spirv/` next to the extension
module, and the loader resolves them relative to that module rather than to the
current directory, so the tests pass from any working directory.

All five kernels are checked against PyTorch by the test suite: attention,
RMSNorm, GEMM, RoPE and the 4-bit dequantizer. They are skipped automatically
when the extension has not been built, so a CPU-only checkout still passes.

## Using Your Own Model

### Step 1: Prepare the dataset

Create a JSONL file with tokenized sequences. Each line must have `input_ids` and `labels`:

```json
{"input_ids": [1, 2, 3, ..., 512], "labels": [1, 2, 3, ..., 512]}
```

To tokenize your own text with the model's tokenizer:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

text = "Your training text here..."
tokens = tokenizer.encode(text)
# Chunk into 512-token segments
for i in range(0, len(tokens) - 512, 512):
    chunk = tokens[i:i+512]
    sample = {"input_ids": chunk, "labels": chunk[1:] + [tokenizer.eos_token_id]}
    # Write sample to JSONL
```

### Step 2: Quantize the expert weights

USAF needs the expert weights in its own 4-bit format: asymmetric per-group
min/max quantization with the scales and zero-points stored in fp16, and the
4-bit values packed two-per-byte. This is implemented in
`usaf/quantization.py` and is not HQQ or any other off-the-shelf scheme, so
the packing cannot be consumed by another library. Reconstruction error on
random fp16 tensors is ~1% relative MSE at a 3.9x compression ratio.

You need to generate the `experts_q4.pt` file for your model:

```python
from usaf.quantization import quantize_4bit
import torch

# Load your model's expert tensors (gate_up_proj and down_proj for each layer)
q_dict = {}
for layer_idx in range(num_layers):
    for param_name in ["gate_up_proj", "down_proj"]:
        # gate_up_proj is fused as [num_experts, 2*intermediate, hidden]
        # down_proj  is [num_experts, hidden, intermediate]
        weights = load_expert_weights(model_path, layer_idx, param_name)
        q4_entry = quantize_4bit(weights, group_size=128)
        q_dict[f"model.layers.{layer_idx}.mlp.experts.{param_name}"] = q4_entry

torch.save(q_dict, "my-model-q4/experts_q4.pt")
```

### Step 3: Configure and run

```bash
# 'export' is required: without it these are shell-local variables and
# train.py will not see them.
export QUANT_PATH="checkpoints/my-model-q4"   # directory holding experts_q4.pt
export DATASET_PATH="data/train_dataset_12h.jsonl"
export HELDOUT_PATH="data/eval_heldout_12h.jsonl"
export TRAIN_FROM=36                 # first trainable layer (top layers train)
export STEPS=360                     # 2 epochs for ~190K tokens
export FRAC=0.005                    # 0.5% of expert weights active
export MICROBATCH=2                  # sequences per micro-batch

python train.py
```

### Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `SRC_MODEL` | `Qwen3-30B-A3B` | Base model directory or HF id |
| `QUANT_PATH` | `checkpoints/<SRC>_q4` | **Directory** containing `experts_q4.pt` |
| `DATASET_PATH` | `data/train_dataset_12h.jsonl` | JSONL file with training samples |
| `HELDOUT_PATH` | `data/eval_heldout_12h.jsonl` | JSONL file with held-out samples |
| `TRAIN_FROM` | 40 | First trainable layer (0-39 are frozen). Clamped to the model's real depth. |
| `FRAC` | 0.005 | Fraction of expert weights active (0.5%) |
| `STEPS` | 180 | Training steps |
| `EPOCHS` | 0 | If > 0, recompute `STEPS` from the dataset size |
| `MICROBATCH` | 4 | Sequences per micro-batch |
| `ACCUM` | 1 | Gradient accumulation; effective batch = `MICROBATCH` x `ACCUM` |
| `LR_PEAK` | 2e-4 | Peak learning rate (cosine decay) |
| `RESELECT_EVERY` | 50 | RigL reselection frequency (in steps) |
| `RESELECT_DROP` | 0.1 | Fraction of active elements eligible to be swapped |
| `EVAL_EVERY` | 15 | Evaluate every N steps |
| `SKIP_FINAL_EVAL` | 0 | Set to 1 to skip the final eval; results are then written as `null`, never as a number |
| `USE_CUDA` | 0 | Set to `1` for NVIDIA GPUs |
| `USE_AMP` | 1 | cuDNN benchmark + TF32 (CUDA only) |
| `USE_MULTI_GPU` | 1 | DataParallel (CUDA only) |
| `USE_VK` | 0 | Vulkan attention path |
| `USE_VK_DEQUANT` | 0 | Vulkan GPU dequant (off by default: slower than CPU here) |
| `FROZEN_CACHE_N` | 0 | Number of samples to cache (0 = all) |

### Supported GPU Configurations

| Setup | Command |
|---|---|
| AMD GPU (RX 6000/7000) | `python train.py` |
| NVIDIA single GPU | `USE_CUDA=1 python train.py` |
| NVIDIA dual GPU | `USE_CUDA=1 USE_MULTI_GPU=1 python train.py` |
| CPU fallback | `python train.py` (automatic) |

The CPU fallback is real: if `torch-directml` is missing, or present but built
against a different torch than the installed one, the run prints why and
continues on CPU instead of dying on the import. It is much slower - the sparse
mechanism is unchanged, only the device differs.

`TRAIN_FROM` is clamped to the model's real depth. If the quantized weights do
not cover the requested range, the run trains the layers that are present and
says so rather than failing.

### Troubleshooting

**"CUDA out of memory"**: Reduce `MICROBATCH` to 1 or increase `TRAIN_FROM` to freeze more layers.

**"No module named torch_directml"** on NVIDIA: Expected. The code auto-detects and uses CUDA. Set `USE_CUDA=1`.

**Loss not decreasing**: Ensure `FRAC` is high enough (>0.001). Try 2-3 epochs with `EPOCHS=3`. Check dataset quality.

**Frozen cache takes too long**: Set `FROZEN_CACHE_N=50` to only cache the first 50 samples. Or disable with `USE_FROZEN_CACHE=0`.

## Future Work

- Benchmarks against LoRA/QLoRA/DoRA on A100-class hardware
- Full Vulkan attention pipeline for cross-vendor acceleration
- Distributed training (FSDP)
- Tests on DeepSeek-V4 Pro, Kimi K2.5, Mistral Large 3 — need hardware

## License

Apache 2.0. [LICENSE](LICENSE). Contributions: [CLA](CLA.md).
