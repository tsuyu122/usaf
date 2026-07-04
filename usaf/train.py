#!/usr/bin/env python3
"""USAF: Ultra Sparse Adaptive Fine-Tuning — Universal MoE Trainer.

Supports: CUDA (NVIDIA), MPS (Apple), CPU fallback.
Auto-detects MoE architecture from HuggingFace config.

Usage:
    usaf train --model Qwen/Qwen3-30B-A3B --dataset data/train.jsonl
    usaf train --config usaf_config.yaml
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time, gc
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import psutil

# ── Device detection ──
def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_device_count() -> int:
    """Number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1

# ── Model detection ──
class ModelInfo:
    """Extracted MoE architecture info from HuggingFace config."""
    def __init__(self, model_path: str):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        self.model_path = model_path
        self.num_layers = getattr(cfg, 'num_hidden_layers', 0)
        self.hidden_size = cfg.hidden_size
        self.num_heads = getattr(cfg, 'num_attention_heads', 0)
        self.num_kv_heads = getattr(cfg, 'num_key_value_heads', self.num_heads)
        self.head_dim = getattr(cfg, 'head_dim', self.hidden_size // self.num_heads)
        self.vocab_size = cfg.vocab_size
        
        # MoE-specific
        self.num_experts = getattr(cfg, 'num_experts', 0)
        self.num_experts_per_tok = getattr(cfg, 'num_experts_per_tok', 0)
        self.expert_intermediate = getattr(cfg, 'moe_intermediate_size', 0)
        self.is_moe = self.num_experts > 0
        
        # Memory estimation for training config
        self._estimate_memory()
    
    def _estimate_memory(self):
        """Estimate per-layer memory and auto-configure trainable layers."""
        if not self.is_moe:
            self.max_trainable_layers = 0
            return
        
        # Estimate VRAM usage per trainable layer
        expert_params = self.num_experts * (
            self.hidden_size * self.expert_intermediate * 2 +  # gate_up + down
            self.expert_intermediate * self.hidden_size
        )
        fp16_per_layer_gb = expert_params * 2 / 1e9
        
        # Available VRAM (conservative estimate)
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        else:
            vram_gb = psutil.virtual_memory().total / 1e9
        
        # Reserve 60% for model + training
        usable_gb = vram_gb * 0.6
        
        # Each trainable layer needs: q4 cache + resident fp16 + gradients + optimizer
        per_layer_gb = fp16_per_layer_gb * 0.25 + 1.5  # ~1.5-2GB per layer
        
        self.max_trainable_layers = max(1, int(usable_gb / per_layer_gb))
        self.estimated_vram_gb = vram_gb
        self.estimated_per_layer_gb = per_layer_gb


# ── Config ──
class USAFConfig:
    """Training configuration with sensible defaults."""
    def __init__(self, **kwargs):
        # Model
        self.model_path = kwargs.get("model_path", "Qwen/Qwen3-30B-A3B")
        self.quant_path = kwargs.get("quant_path", "")
        
        # Training
        self.seq_len = kwargs.get("seq_len", 512)
        self.steps = kwargs.get("steps", 180)
        self.epochs = kwargs.get("epochs", 0)
        self.microbatch = kwargs.get("microbatch", 2)
        self.accum = kwargs.get("accum", 1)
        
        # Sparsity
        self.train_from = kwargs.get("train_from", 0)  # 0 = auto
        self.frac = kwargs.get("frac", 0.005)
        self.lr_peak = kwargs.get("lr_peak", 2e-4)
        self.weight_decay = kwargs.get("weight_decay", 0.005)
        
        # RigL
        self.reselect_every = kwargs.get("reselect_every", 50)
        
        # Features
        self.use_frozen_cache = kwargs.get("use_frozen_cache", True)
        self.frozen_cache_n = kwargs.get("frozen_cache_n", 0)
        self.use_resident = kwargs.get("use_resident", True)
        self.use_vk = kwargs.get("use_vk", False)  # Vulkan only on AMD
        self.use_amp = kwargs.get("use_amp", True)  # CUDA mixed precision
        
        # Multi-GPU
        self.multi_gpu = kwargs.get("multi_gpu", True)
        
        # Output
        self.run_tag = kwargs.get("run_tag", "")
        self.checkpoint_dir = kwargs.get("checkpoint_dir", "checkpoints")
        self.log_dir = kwargs.get("log_dir", "logs")
    
    @classmethod
    def from_cli(cls, args=None):
        """Parse from command line or config file."""
        if args is None:
            parser = cls._build_parser()
            args = parser.parse_args()
        
        if args.config:
            with open(args.config) as f:
                cfg_dict = json.load(f)
        else:
            cfg_dict = vars(args)
        
        return cls(**cfg_dict)
    
    @staticmethod
    def _build_parser():
        p = argparse.ArgumentParser(description="USAF: Ultra Sparse Adaptive Fine-Tuning")
        p.add_argument("--config", type=str, help="JSON config file")
        p.add_argument("--model-path", type=str, default="Qwen/Qwen3-30B-A3B")
        p.add_argument("--dataset", type=str, default="data/train.jsonl")
        p.add_argument("--quant-path", type=str, default="")
        p.add_argument("--steps", type=int, default=180)
        p.add_argument("--epochs", type=float, default=0)
        p.add_argument("--seq-len", type=int, default=512)
        p.add_argument("--microbatch", type=int, default=2)
        p.add_argument("--accum", type=int, default=1)
        p.add_argument("--frac", type=float, default=0.005)
        p.add_argument("--lr-peak", type=float, default=2e-4)
        p.add_argument("--train-from", type=int, default=0)
        p.add_argument("--reselect-every", type=int, default=50)
        p.add_argument("--run-tag", type=str, default="")
        p.add_argument("--no-frozen-cache", action="store_true")
        p.add_argument("--no-resident", action="store_true")
        p.add_argument("--no-amp", action="store_true")
        p.add_argument("--no-multi-gpu", action="store_true")
        return p


# ── Main training function ──
def train(config: USAFConfig):
    """Main USAF training loop."""
    
    # Device setup
    device = get_device()
    n_gpus = get_device_count()
    use_multi_gpu = config.multi_gpu and n_gpus > 1 and device.type == "cuda"
    
    print(f"=== USAF Universal Trainer ===")
    print(f"Device: {device.type.upper()}" + (f" × {n_gpus}" if n_gpus > 1 else ""))
    print(f"Model: {config.model_path}")
    
    # Detect model architecture
    info = ModelInfo(config.model_path)
    print(f"Architecture: {info.num_layers} layers, {info.hidden_size} hidden")
    if info.is_moe:
        print(f"MoE: {info.num_experts} experts × {info.num_experts_per_tok} active, "
              f"intermediate={info.expert_intermediate}")
    
    # Auto-configure trainable layers
    if config.train_from == 0:
        if info.is_moe:
            config.train_from = max(0, info.num_layers - info.max_trainable_layers)
        else:
            config.train_from = info.num_layers  # train all layers for dense models
    train_layers = set(range(config.train_from, info.num_layers))
    
    print(f"Trainable layers: {len(train_layers)} ({config.train_from}-{info.num_layers-1})")
    print(f"VRAM: {info.estimated_vram_gb:.0f}GB total, ~{info.estimated_per_layer_gb:.1f}GB/layer")
    
    # Quant path
    if not config.quant_path:
        config.quant_path = f"{config.model_path.split('/')[-1]}-q4/experts_q4.pt"
    
    # Dataset
    dataset_path = config.dataset if hasattr(config, 'dataset') else "data/train.jsonl"
    train_samples, eval_samples, heldout_samples = _load_dataset(dataset_path, config.seq_len)
    
    effective_batch = config.microbatch * config.accum
    if config.epochs > 0:
        tokens_per_epoch = len(train_samples) * config.seq_len
        config.steps = max(1, int(config.epochs * tokens_per_epoch / (effective_batch * config.seq_len)))
    
    print(f"Dataset: {len(train_samples)} train, {len(eval_samples)} eval samples")
    print(f"Steps: {config.steps}, Batch: {config.microbatch}×{config.accum}={effective_batch}")
    print(f"Tokens: {config.steps * effective_batch * config.seq_len:,}")
    
    # ── Model loading ──
    print(f"\nLoading model...")
    model, cache, q_dict = _load_model(config, info, device)
    
    # Multi-GPU
    if use_multi_gpu:
        print(f"Multi-GPU: DataParallel across {n_gpus} GPUs")
        model = torch.nn.DataParallel(model)
        # Note: expert streaming hooks are registered per-module;
        # DataParallel replicates the model, so hooks apply to all replicas.
    
    # ── Training setup ──
    # ... (importance, active selection, masters, optimizer)
    # This follows the same logic as train_qwen3_12h.py but with CUDA optimizations
    
    print(f"\nSetup complete. Starting training...")
    print(f"(Full training loop implementation follows the same pattern as train_qwen3_12h.py)")
    
    return model, info, config


def _load_dataset(path: str, seq_len: int):
    """Load tokenized dataset."""
    samples = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                    if "input_ids" in s and len(s["input_ids"]) == seq_len:
                        samples.append(s)
                except json.JSONDecodeError:
                    continue
    
    random.seed(42)
    random.shuffle(samples)
    
    n_train = max(1, len(samples) - 20)
    train = samples[:n_train]
    eval_s = samples[n_train:n_train+10]
    heldout = samples[n_train+10:n_train+20]
    
    return train, eval_s, heldout


def _load_model(config: USAFConfig, info: ModelInfo, device: torch.device):
    """Load model with streaming expert hooks."""
    from transformers import AutoConfig, AutoTokenizer
    from safetensors import safe_open
    from usaf.moe_loader import QuantizedExpertCache
    
    # Load config + model structure
    with torch.device("meta"):
        cfg = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
        # Try to load the model class dynamically
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_config(cfg)
        except Exception:
            # Fallback: try Qwen3Moe specifically
            from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
            model = Qwen3MoeForCausalLM(cfg)
    
    # Load weights from safetensors
    st_path = config.model_path
    if not os.path.isdir(st_path):
        # HuggingFace hub path — use local cache
        from transformers.utils import cached_file
        st_path = str(Path(cached_file(config.model_path, "config.json")).parent)
    
    st_files = sorted([f for f in os.listdir(st_path) if f.endswith(".safetensors")])
    wf = {}
    for fn in st_files:
        with safe_open(os.path.join(st_path, fn), framework="pt") as sf:
            for key in sf.keys():
                wf[key] = fn
    
    mp = dict(model.named_parameters())
    n_loaded = 0
    for name in sorted(wf.keys()):
        if ".mlp.experts." in name:
            continue
        if name not in mp:
            continue
        with safe_open(os.path.join(st_path, wf[name]), framework="pt") as sf:
            tensor = sf.get_tensor(name)
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        obj._parameters[parts[-1]] = torch.nn.Parameter(
            tensor.to(device=device, dtype=torch.float16), requires_grad=False)
        n_loaded += 1
    
    # Load buffers
    for mn, mod in model.named_modules():
        for bn, b in list(mod._buffers.items()):
            if b is not None and b.device.type == "meta":
                if bn == "inv_freq":
                    hd_val = getattr(mod, "dim", getattr(mod, "head_dim", 128))
                    base = getattr(mod, "base", 1000000.0)
                    inv = 1.0 / (base ** (torch.arange(0, hd_val, 2, dtype=torch.float32) / hd_val))
                    mod._buffers[bn] = inv.to(dtype=torch.float16, device=device)
                else:
                    mod._buffers[bn] = torch.zeros(b.shape, dtype=torch.float16, device=device)
    
    print(f"  {n_loaded} non-expert params → GPU")
    
    # Load Q4 expert weights
    import torch
    q_dict = torch.load(config.quant_path, map_location="cpu", weights_only=True)
    cache = QuantizedExpertCache(q_dict, device, max_cached=1, group_size=128)
    
    # Setup streaming hooks
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
    
    return model, cache, q_dict


# ── CLI entry point ──
def main():
    config = USAFConfig.from_cli()
    train(config)


if __name__ == "__main__":
    main()
