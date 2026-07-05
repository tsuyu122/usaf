#!/usr/bin/env python3
"""
USAF Kaggle Training Script
============================
Runs on Kaggle's 2x T4 GPUs (CUDA, 16GB each).
Auto-detects MoE architecture, configures sparsity, handles multi-GPU.

Setup (run once):
    pip install transformers safetensors psutil accelerate
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Usage:
    python usaf_kaggle_train.py
"""

import os, sys, json, time, math, random, gc, subprocess, faulthandler, traceback
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

# Tee ALL output to /kaggle/working/run.log (saved as kernel output even if the
# notebook log capture fails or the container is OOM-killed; line-buffered so we
# see up to the point of death).
os.makedirs("/kaggle/working", exist_ok=True)
_logf = open("/kaggle/working/run.log", "w", buffering=1)
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams:
            try: s.write(d); s.flush()
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass
    def fileno(self):
        return self.streams[0].fileno()
    def isatty(self):
        return False
sys.stdout = _Tee(sys.__stdout__, _logf)
sys.stderr = _Tee(sys.__stderr__, _logf)
faulthandler.enable(sys.__stderr__)

def _ram():
    import psutil as _p
    m = _p.virtual_memory()
    return f"RAM {m.used/1e9:.1f}/{m.total/1e9:.1f}GB used"

print("=== boot ===", flush=True)
for _lim in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
        v = open(_lim).read().strip()
        gb = int(v)/1e9 if v.isdigit() else v
        print(f"cgroup RAM limit ({_lim}): {gb}", flush=True); break
    except Exception:
        pass

# Pin transformers to match the model's fused-expert architecture (Qwen3MoeExperts).
# Kaggle ships an older transformers where experts are a ModuleList, which breaks
# both weight loading and the grad-capture patch.
print("pip install transformers==5.12.1 ...", flush=True)
_pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                       "transformers==5.12.1", "safetensors"],
                      capture_output=True, text=True)
print(f"pip rc={_pip.returncode}", flush=True)
if _pip.returncode != 0:
    print("PIP STDERR:", _pip.stderr[-2000:], flush=True)

# Locate the usaf package under /kaggle/input (mount slug/path varies) and add
# its PARENT to sys.path so `import usaf` works.
_usaf_parent = None
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if os.path.basename(_root) == "usaf" and "__init__.py" in _files:
        _usaf_parent = os.path.dirname(_root)
        break
if _usaf_parent:
    sys.path.insert(0, _usaf_parent)
    print(f"usaf package parent: {_usaf_parent}", flush=True)
else:
    print("WARNING: usaf package not found under /kaggle/input", flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import psutil

# ===============================================
# CONFIG - modify these for different runs
# ===============================================

# Auto-resolve dataset mount paths (Kaggle slug may differ from what we expect).
_INP = "/kaggle/input"
print("INPUT dirs:", os.listdir(_INP) if os.path.isdir(_INP) else "NONE", flush=True)

def _find_dir_with(pred, what):
    for root, _dirs, files in os.walk(_INP):
        if pred(root, files):
            print(f"resolved {what}: {root}", flush=True)
            return root
    raise FileNotFoundError(f"could not resolve {what} under {_INP}")

def _find_file(fname, what):
    for root, _dirs, files in os.walk(_INP):
        if fname in files:
            p = os.path.join(root, fname)
            print(f"resolved {what}: {p}", flush=True)
            return p
    raise FileNotFoundError(f"could not find {fname} under {_INP}")

# base = dir with qwen3_moe config.json AND actual .safetensors weights
# (the q4 dataset also has a qwen3_moe config.json but only .pt / .index.json).
def _is_base(root, files):
    if "config.json" not in files:
        return False
    if not any(f.endswith(".safetensors") for f in files):
        return False
    try:
        return json.load(open(os.path.join(root, "config.json"))).get("model_type") == "qwen3_moe"
    except Exception:
        return False

MODEL_PATH = _find_dir_with(_is_base, "MODEL_PATH (base)")
QUANT_PATH = _find_file("experts_q4.pt", "QUANT_PATH")
DATASET_PATH = _find_file("train_dataset.jsonl", "DATASET_PATH")

# BENCHMARK config: short run to measure real CUDA tok/s + loss trend.
STEPS = 20           # ~5 epochs over the pool -> loss should visibly drop
POOL_N = 4           # frozen precompute is ~140s/sample; keep the pool tiny
SEQ_LEN = 256        # 14.5GB T4 OOMs at 512 with dense 128-expert compute
MICROBATCH = 1
ACCUM = 1
FRAC = 0.005         # 0.5% sparsity
LR_PEAK = 2e-4
RESELECT_EVERY = 50
TRAIN_FROM = 44      # 4 trainable layers (44-47): resident cache ~5GB (32GB cgroup)
USE_AMP = True       # Automatic Mixed Precision (CUDA)
# The training loop runs the decoder layers manually (per-layer fwd/bwd), which
# bypasses DataParallel - so this measures a single T4 honestly. 2x T4 speedup
# would need a real model/pipeline-parallel rewrite (future work).
MULTI_GPU = False
RUN_TAG = "cuda_bench"
WARMUP_SKIP = 3      # steps excluded from the reported tok/s average

# ===============================================
# DEVICE SETUP
# ===============================================

assert torch.cuda.is_available(), "CUDA required!"
n_gpus = torch.cuda.device_count()
device = torch.device("cuda")
print(f"CUDA devices: {n_gpus}")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_memory/1e9:.1f}GB VRAM")

if MULTI_GPU and n_gpus > 1:
    print(f"Multi-GPU: DataParallel across {n_gpus} GPUs")
else:
    MULTI_GPU = False
    print("Single GPU mode")

# ===============================================
# MODEL LOADING
# ===============================================

print(f"\n=== Loading model === ({_ram()})")
import transformers
print(f"transformers {transformers.__version__}")
from transformers import AutoConfig
from safetensors import safe_open

cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
H = cfg.hidden_size
nH = cfg.num_attention_heads
nKV = getattr(cfg, 'num_key_value_heads', nH)
hd = getattr(cfg, 'head_dim', H // nH)
N_LAYERS = cfg.num_hidden_layers
N_EXPERTS = getattr(cfg, 'num_experts', 0)
EXP_ACTIVE = getattr(cfg, 'num_experts_per_tok', 0)
EXP_INT = getattr(cfg, 'moe_intermediate_size', 0)
is_moe = N_EXPERTS > 0

print(f"Model: {N_LAYERS} layers, H={H}, heads={nH}/{nKV}, hd={hd}")
if is_moe:
    print(f"MoE: {N_EXPERTS} experts, {EXP_ACTIVE} active, intermediate={EXP_INT}")

# Auto-configure trainable layers
if TRAIN_FROM == 0:
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    per_layer_gb = 2.0  # rough estimate
    max_trainable = max(1, int(vram_gb * 0.5 / per_layer_gb))
    TRAIN_FROM = max(0, N_LAYERS - max_trainable)

TRAIN_LAYERS = set(range(TRAIN_FROM, N_LAYERS))
print(f"Trainable: {len(TRAIN_LAYERS)} layers ({TRAIN_FROM}-{N_LAYERS-1})")

# Load model structure
with torch.device("meta"):
    from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
    model = Qwen3MoeForCausalLM(cfg)

# Load weights
st_files = sorted([f for f in os.listdir(MODEL_PATH) if f.endswith(".safetensors")])
wf = {}
for fn in st_files:
    with safe_open(os.path.join(MODEL_PATH, fn), framework="pt") as sf:
        for key in sf.keys():
            wf[key] = fn

mp = dict(model.named_parameters())
for name in sorted(wf.keys()):
    if ".mlp.experts." in name:
        continue
    if name not in mp:
        continue
    with safe_open(os.path.join(MODEL_PATH, wf[name]), framework="pt") as sf:
        tensor = sf.get_tensor(name)
    parts = name.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    obj._parameters[parts[-1]] = nn.Parameter(
        tensor.to(device=device, dtype=torch.float16), requires_grad=False)

# Load buffers
for mn, mod in model.named_modules():
    for bn, b in list(mod._buffers.items()):
        if b is not None and b.device.type == "meta":
            if bn == "inv_freq":
                hd_val = getattr(mod, "dim", getattr(mod, "head_dim", hd))
                base = getattr(mod, "base", 1000000.0)
                inv = 1.0 / (base ** (torch.arange(0, hd_val, 2, dtype=torch.float32) / hd_val))
                mod._buffers[bn] = inv.to(dtype=torch.float16, device=device)

print(f"Model loaded on {device}")

# Apply the dense-masked MoE forward + per-expert grad-capture protocol.
# This is what routes sparse grads into the store via `module._grad_capture`.
# Without it the native HF forward runs but sparse grads stay zero (no training).
# Same dense-masked path as the DML baseline -> apples-to-apples tok/s.
from usaf.qwen3moe_dml import patch_qwen3moe_for_dml
patch_qwen3moe_for_dml()
print("Applied qwen3moe grad-capture forward patch")

# ===============================================
# Q4 EXPERT CACHE
# ===============================================

from usaf.moe_loader import QuantizedExpertCache

print(f"Loading q4 dict ({_ram()}) ...", flush=True)
q_dict = torch.load(QUANT_PATH, map_location="cpu", weights_only=True)
print(f"q4 dict loaded: {len(q_dict)} entries ({_ram()})", flush=True)
cache = QuantizedExpertCache(q_dict, device, max_cached=1, group_size=128)

# Expert streaming hooks
for mname, mod in model.named_modules():
    if not mname.endswith(".mlp.experts"):
        continue
    mod._parameters.clear()
    if hasattr(mod, '_buffers'):
        mod._buffers.clear()
    
    def make_pre(name):
        def pre(module, args):
            weights = cache.get_expert_weights(name)
            for pn, param in weights.items():
                module._parameters[pn] = param
        return pre
    
    def make_post():
        def post(module, args, output):
            module._parameters.clear()
            return output
        return post
    
    mod.register_forward_pre_hook(make_pre(mname))
    mod.register_forward_hook(make_post())

# Multi-GPU
if MULTI_GPU:
    model = nn.DataParallel(model)
    print("DataParallel enabled")

# ===============================================
# DATASET
# ===============================================

print("\n=== Loading dataset ===")
with open(DATASET_PATH) as f:
    all_samples = [json.loads(line) for line in f if line.strip()]

random.seed(42)
random.shuffle(all_samples)
train_samples = all_samples[:-20]
eval_samples = all_samples[-20:-10]
heldout_samples = all_samples[-10:]

for i, s in enumerate(train_samples):
    s["_fidx"] = i

eff_batch = MICROBATCH * ACCUM
tokens_total = STEPS * eff_batch * SEQ_LEN
print(f"Train: {len(train_samples)}, Eval: {len(eval_samples)}, Heldout: {len(heldout_samples)}")
print(f"Tokens: {tokens_total:,} over {STEPS} steps")

# ===============================================
# CUDA OPTIMIZATIONS
# ===============================================

if USE_AMP:
    scaler = torch.cuda.amp.GradScaler()
    print("AMP (Automatic Mixed Precision) enabled")
else:
    scaler = None

# CUDA benchmarking optimization
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("CUDA optimizations: cudnn.benchmark + tf32 enabled")

# ===============================================
# PRELUDE FUNCTIONS
# ===============================================

def ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3

def _prelude(input_ids):
    hidden = model.module.embed_tokens(input_ids) if MULTI_GPU else model.model.embed_tokens(input_ids)
    seq_len = hidden.shape[1]
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    rotary_emb = model.module.model.rotary_emb if MULTI_GPU else model.model.rotary_emb
    cos, sin = rotary_emb(hidden, position_ids=pos_ids)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(torch.float16).min, device=device, dtype=torch.float16),
        diagonal=1).unsqueeze(0).unsqueeze(0)
    return hidden, pos_ids, (cos, sin), causal_mask

def _head_loss(hidden, labels):
    norm = model.module.model.norm if MULTI_GPU else model.model.norm
    lm_head = model.module.lm_head if MULTI_GPU else model.lm_head
    hidden = norm(hidden)
    logits = lm_head(hidden)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1))

# ===============================================
# TRAINING SETUP (simplified - full version in train_qwen3_12h.py)
# ===============================================

print("\n=== Training setup ===")
from usaf.sparse_optim import SparseAdam
from usaf.quantization import dequantize_4bit

DETACH_AT = TRAIN_FROM - 1
_train_names = [f"model.layers.{li}.mlp.experts.{pn}"
                for li in sorted(TRAIN_LAYERS) for pn in ("gate_up_proj", "down_proj")]

def _q_shape(fn):
    e = q_dict[fn]
    # Handle both dict and tuple formats
    if isinstance(e, dict):
        return tuple(e["shape"])
    return tuple(e[3])

_shapes = {fn: _q_shape(fn) for fn in _train_names}

# Importance phase
from usaf.moe_loader import TopKImportanceStore
imp_store = TopKImportanceStore(_shapes, frac=FRAC)

# Register grad capture hooks
for mname, mod in model.named_modules():
    if not mname.endswith(".mlp.experts"):
        continue
    mod._grad_capture = (imp_store, mname)

# Simple fwd_bwd for importance
def _qwen_layer(layer_idx):
    layers = model.module.model.layers if MULTI_GPU else model.model.layers
    return layers[layer_idx]

# ---- Frozen activation cache (roadmap opt #1) --------------------------------
# The 15GB q4 dict + dequant peak exceeds the Kaggle kernel RAM cgroup limit.
# Frozen layers (0..TRAIN_FROM-1) are deterministic under no_grad, so we run
# them ONCE per sample, cache hidden@TRAIN_FROM, then free the frozen q4 entries
# (~12GB). Training then streams only the trainable layers -> fits comfortably.
# Also makes the per-step tok/s reflect the trainable path (the real steady cost
# once frozen activations are cached).
DETACH_AT = TRAIN_FROM - 1

# Fixed rotary/mask for the trainable layers (positions/seq_len are constant).
_pos_ids_t = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
_rotary = model.model.rotary_emb
_dummy = torch.zeros(1, SEQ_LEN, H, device=device, dtype=torch.float16)
_cos_t, _sin_t = _rotary(_dummy, position_ids=_pos_ids_t)
_pe_t = (_cos_t, _sin_t)
_mask_t = torch.triu(
    torch.full((SEQ_LEN, SEQ_LEN), torch.finfo(torch.float16).min, device=device, dtype=torch.float16),
    diagonal=1).unsqueeze(0).unsqueeze(0)
del _dummy

def _frozen_hidden(sample):
    ids = torch.tensor(sample["input_ids"][:SEQ_LEN], dtype=torch.long).unsqueeze(0).to(device)
    hidden, pos_ids, pe, mask = _prelude(ids)
    with torch.no_grad():
        for i in range(TRAIN_FROM):
            hidden = _qwen_layer(i)(hidden, attention_mask=mask, position_ids=pos_ids, position_embeddings=pe)
        cache.evict_all()
    return hidden.detach().to("cpu")  # input to layer TRAIN_FROM

def _step_cached(item, bwd_scale=1.0):
    # Trainable layers only, on a cached frozen hidden. Per-layer detached
    # backward -> grad-capture hooks fire into whatever store is wired.
    h0 = item["hidden"].to(device)
    lbl = torch.tensor(item["labels"][:SEQ_LEN], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        xs = []
        hidden = h0
        for i in range(TRAIN_FROM, N_LAYERS):
            xs.append(hidden)
            hidden = _qwen_layer(i)(hidden, attention_mask=_mask_t, position_ids=_pos_ids_t, position_embeddings=_pe_t)
        cache.evict_all()
    h_last = hidden.detach().requires_grad_(True)
    loss = _head_loss(h_last, lbl)
    # Single, consistent loss scaling (bwd_scale). Do NOT also use the AMP
    # GradScaler here: double-scaling (65536 * manual) overflowed fp16 grads to
    # inf, so every optimizer step got skipped.
    (loss * bwd_scale).backward()
    g = h_last.grad
    for j in range(len(xs) - 1, -1, -1):
        i = TRAIN_FROM + j
        x = xs[j].detach().requires_grad_(True)
        out = _qwen_layer(i)(x, attention_mask=_mask_t, position_ids=_pos_ids_t, position_embeddings=_pe_t)
        out.backward(g)
        g = x.grad
        cache.evict_all()
    return loss.item()

# Precompute frozen hidden for the benchmark sample pool.
POOL = train_samples[:POOL_N]
print(f"\n=== Frozen precompute ({len(POOL)} samples) === ({_ram()})", flush=True)
_t0 = time.time()
frozen_cache = []
for _k, _s in enumerate(POOL):
    frozen_cache.append({"hidden": _frozen_hidden(_s), "labels": _s["labels"]})
    if _k % 5 == 0:
        print(f"  frozen {_k+1}/{len(POOL)} | {time.time()-_t0:.0f}s | {_ram()}", flush=True)
_frozen_secs = time.time() - _t0
print(f"Frozen precompute done: {len(frozen_cache)} samples in {_frozen_secs:.0f}s "
      f"({_frozen_secs/len(POOL):.1f}s/sample)", flush=True)

# Free frozen-layer q4 entries -> big RAM drop.
cache.free_frozen(DETACH_AT)
gc.collect()
print(f"Freed frozen q4 (layers 0..{DETACH_AT}) ({_ram()})", flush=True)

# Run importance on cached frozen hidden (trainable layers only).
print("Importance phase...")
t0 = time.time()
for imp_i in range(3):
    loss_imp = _step_cached(frozen_cache[imp_i % len(frozen_cache)])
    print(f"  imp {imp_i+1}/3 | loss {loss_imp:.4f} | {time.time()-t0:.0f}s | {_ram()}")

active_idx = imp_store.select(FRAC)
ta = sum(i.numel() for i in active_idx.values())
te = sum(math.prod(_shapes[fn]) for fn in active_idx)
print(f"Active: {ta:,}/{te:,} ({100*ta/max(te,1):.4f}%)")

# Masters + overlays
masters = {}
for fname, aidx in active_idx.items():
    aidx = aidx.reshape(-1).to(torch.long)
    entry = q_dict[fname]
    if isinstance(entry, dict):
        t = dequantize_4bit(entry["q"], entry["s"], entry["z"], entry["shape"], group_size=128)
    else:
        t = dequantize_4bit(entry[0], entry[1], entry[2], entry[3], group_size=128)
    vals = t.reshape(-1).index_select(0, aidx).float()
    del t
    p = nn.Parameter(vals, requires_grad=False)
    masters[fname] = p
    cache.overlays[fname] = (aidx, p)

# SparseGradStore
from usaf.moe_loader import SparseGradStore
sparse_store = SparseGradStore(active_idx, _shapes)
for mname, mod in model.named_modules():
    if not mname.endswith(".mlp.experts"):
        continue
    mod._grad_capture = (sparse_store, mname)

# Resident mode
cache.make_resident(TRAIN_LAYERS)
cache.apply_resident_overlays(active_idx, masters)
cache._prefetch_disabled = True

# Optimizer
print(f"Building optimizer ({_ram()}) ...", flush=True)
opt = SparseAdam(masters, active_idx=active_idx, lr=LR_PEAK, weight_decay=0.005, compact_params=True)

# ===============================================
# TRAINING LOOP
# ===============================================

print(f"\n=== Training ({STEPS} steps) ===")
print(f"Batch: {MICROBATCH}x{ACCUM}={eff_batch}, Tokens/step: {eff_batch*SEQ_LEN}")
print(f"LR: {LR_PEAK}, Sparsity: {FRAC}, Reselect: every {RESELECT_EVERY}")
print(f"RAM: {ram():.1f}GB")

def fwd_bwd(item, zero_store=True, bwd_scale=1.0):
    # Trainable path on a cached frozen hidden (see _step_cached).
    if zero_store:
        sparse_store.zero_()
    return _step_cached(item, bwd_scale=bwd_scale)

def _eval_loss(item):
    with torch.no_grad():
        h = item["hidden"].to(device)
        for i in range(TRAIN_FROM, N_LAYERS):
            h = _qwen_layer(i)(h, attention_mask=_mask_t, position_ids=_pos_ids_t, position_embeddings=_pe_t)
        cache.evict_all()
        lbl = torch.tensor(item["labels"][:SEQ_LEN], dtype=torch.long).unsqueeze(0).to(device)
        return _head_loss(h, lbl).item()

_probe_before = _eval_loss(frozen_cache[0])
print(f"Probe loss (sample 0) BEFORE training: {_probe_before:.4f}", flush=True)

losses = []
step_dts = []
si = 0
t_start = time.time()
pr_peak = 0.0
good_streak = 0
loss_scale = 4096.0

for step in range(1, STEPS + 1):
    t_step = time.time()
    
    # LR schedule: cosine decay
    if step <= max(1, int(STEPS * 0.05)):
        lr = LR_PEAK * step / max(1, int(STEPS * 0.05))
    else:
        progress = (step - max(1, int(STEPS * 0.05))) / max(1, STEPS - max(1, int(STEPS * 0.05)))
        lr = LR_PEAK * 0.1 + LR_PEAK * 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    opt.lr = lr
    
    sparse_store.zero_()
    step_loss = 0.0
    
    for a in range(ACCUM):
        item = frozen_cache[si % len(frozen_cache)]
        si += 1
        loss_val = fwd_bwd(item, zero_store=False, bwd_scale=loss_scale)
        step_loss += loss_val
    
    step_loss /= ACCUM
    
    # Optimizer
    denom = loss_scale * ACCUM
    cg = {n: v / denom for n, v in sparse_store.compact.items()}
    finite = all(torch.isfinite(v).all().item() for v in cg.values())
    
    if finite:
        opt.step(compact_grads=cg)
        cache.sync_resident(active_idx, masters)
        good_streak += 1
        if good_streak % 200 == 0:
            loss_scale = min(loss_scale * 2, 65536.0)
    else:
        loss_scale = max(loss_scale / 2, 64.0)
    
    cache.evict_all()
    losses.append(step_loss)
    
    pr = ram()
    if pr > pr_peak:
        pr_peak = pr
    
    dt = time.time() - t_step
    step_dts.append(dt)
    pct = 100.0 * step / STEPS
    eta_h = (STEPS - step) * dt / 3600
    tok_s = eff_batch * SEQ_LEN / dt
    
    print(f"  [{int(pct//4)*'#'+'-'*25:25s}] {pct:5.1f}% | step {step:03d}/{STEPS} | "
          f"loss {step_loss:.4f} | {tok_s:.0f} tok/s | LR {lr:.1e} | RAM {pr:.1f}G | ETA {eta_h:.1f}h",
          flush=True)

t_total = time.time() - t_start
_probe_after = _eval_loss(frozen_cache[0])
print(f"\n=== Training complete ===")
print(f"Probe loss (sample 0): {_probe_before:.4f} -> {_probe_after:.4f} "
      f"(delta {_probe_after - _probe_before:+.4f})")
print(f"Time: {t_total/3600:.1f}h")
print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (min: {min(losses):.4f})")
print(f"Peak RAM: {pr_peak:.1f}GB")
print(f"Skipped: {STEPS - good_streak} steps")
print(f"Tok/s avg (all): {STEPS * eff_batch * SEQ_LEN / t_total:.1f}")
steady = step_dts[WARMUP_SKIP:] if len(step_dts) > WARMUP_SKIP else step_dts
if steady:
    mean_dt = sum(steady) / len(steady)
    print(f"Steady-state (excl {WARMUP_SKIP} warmup): {mean_dt:.1f}s/step, "
          f"{eff_batch * SEQ_LEN / mean_dt:.1f} tok/s")
    print(f"Extrapolated 180 steps: {180 * mean_dt / 3600:.2f}h")
gpu0 = torch.cuda.get_device_properties(0).name
print(f"GPU: {gpu0} | VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

# Save checkpoint
Path("checkpoints").mkdir(exist_ok=True)
ckpt_path = f"checkpoints/usaf_{RUN_TAG}.pt"
torch.save({
    "step": STEPS,
    "active_idx": active_idx,
    "masters": {n: p.data.clone() for n, p in masters.items()},
    "losses": losses,
    "config": {
        "model": MODEL_PATH,
        "train_from": TRAIN_FROM,
        "frac": FRAC,
        "steps": STEPS,
        "lr_peak": LR_PEAK,
    }
}, ckpt_path)
print(f"Checkpoint saved: {ckpt_path}")
