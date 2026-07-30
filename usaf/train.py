"""USAF: Ultra Sparse Adaptive Fine-Tuning â€” Universal Training CLI.

Supports any MoE model from HuggingFace. Auto-detects architecture and configures training.

Usage:
    usaf train --model Qwen/Qwen3-30B-A3B --dataset data/train.jsonl
    usaf train --model deepseek-ai/DeepSeek-MoE-16B --dataset data.jsonl --steps 360
    python -m usaf.train --help
"""
import argparse, json, math, os, random, sys, time, gc
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import psutil


def ram() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def _get_tokenizer():
    """Get the global tokenizer instance, if available."""
    return getattr(_get_tokenizer, "_instance", None)


def _set_tokenizer(tok):
    _get_tokenizer._instance = tok


def _set_model(model):
    _set_tokenizer(getattr(model, "tokenizer", None))


def build_parser():
    p = argparse.ArgumentParser(description="USAF: Ultra Sparse Adaptive Fine-Tuning")
    
    p.add_argument("--model", type=str, required=True,
                   help="HuggingFace model ID or local path")
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to JSONL dataset file")
    
    p.add_argument("--quant-path", type=str, default="",
                   help="Path to q4 experts file. Auto-detected if empty.")
    
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--epochs", type=float, default=0,
                   help="If >0, overrides --steps based on dataset size")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--accum", type=int, default=1)
    
    p.add_argument("--frac", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.005)
    p.add_argument("--train-from", type=int, default=0,
                   help="First trainable layer (0=auto)")
    p.add_argument("--reselect-every", type=int, default=50)
    
    p.add_argument("--no-frozen-cache", action="store_true")
    p.add_argument("--no-resident", action="store_true")
    p.add_argument("--frozen-cache-n", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=15)
    
    p.add_argument("--cuda", action="store_true", default=None)
    p.add_argument("--no-cuda", action="store_true", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-multi-gpu", action="store_true")
    
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--log-dir", type=str, default="logs")
    
    p.add_argument("--resume", type=str, default="",
                   help="Path to sparse checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=50,
                   help="Save checkpoint every N steps (0=no mid-training saves)")
    p.add_argument("--export", type=str, default="",
                   help="Export merged weights path (e.g. experts_finetuned_q4.pt)")
    
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, only evaluate model/checkpoint")
    p.add_argument("--eval-datasets", type=str, default="synthetic-cpp",
                   help="Comma-separated list of eval datasets")
    p.add_argument("--eval-samples", type=int, default=64,
                   help="Max samples per eval dataset")
    p.add_argument("--eval-report", type=str, default="",
                   help="Path to save eval report JSON")
    
    return p


@dataclass
class TrainConfig:
    model_path: str
    dataset_path: str
    quant_path: str = ""
    steps: int = 180
    epochs: float = 0
    seq_len: int = 512
    microbatch: int = 2
    accum: int = 1
    frac: float = 0.005
    lr_peak: float = 2e-4
    weight_decay: float = 0.005
    train_from: int = 0
    reselect_every: int = 50
    use_frozen_cache: bool = True
    frozen_cache_n: int = 0
    use_resident: bool = True
    eval_every: int = 15
    use_cuda: Optional[bool] = None
    use_amp: bool = True
    use_multi_gpu: bool = True
    tag: str = ""
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    resume_path: str = ""
    save_every: int = 50
    export_path: str = ""
    eval_only: bool = False
    eval_datasets: str = "synthetic-cpp"
    eval_samples: int = 64
    eval_report: str = ""


def parse_args(args=None) -> TrainConfig:
    p = build_parser()
    ns = p.parse_args(args)
    
    use_cuda = ns.cuda
    if use_cuda is None and ns.no_cuda:
        use_cuda = False
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    
    return TrainConfig(
        model_path=ns.model,
        dataset_path=ns.dataset,
        quant_path=ns.quant_path,
        steps=ns.steps,
        epochs=ns.epochs,
        seq_len=ns.seq_len,
        microbatch=ns.microbatch,
        accum=ns.accum,
        frac=ns.frac,
        lr_peak=ns.lr,
        weight_decay=ns.wd,
        train_from=ns.train_from,
        reselect_every=ns.reselect_every,
        use_frozen_cache=not ns.no_frozen_cache,
        frozen_cache_n=ns.frozen_cache_n,
        use_resident=not ns.no_resident,
        eval_every=ns.eval_every,
        use_cuda=use_cuda,
        use_amp=not ns.no_amp,
        use_multi_gpu=not ns.no_multi_gpu,
        tag=ns.tag,
        checkpoint_dir=ns.checkpoint_dir,
        log_dir=ns.log_dir,
        resume_path=ns.resume,
        save_every=ns.save_every,
        export_path=ns.export,
        eval_only=ns.eval_only,
        eval_datasets=ns.eval_datasets,
        eval_samples=ns.eval_samples,
        eval_report=ns.eval_report,
    )


def setup_device(config: TrainConfig) -> Tuple[torch.device, int, object]:
    """Configure device, AMP scaler, and multi-GPU."""
    from usaf.qwen3moe_dml import patch_qwen3moe_for_dml
    patch_qwen3moe_for_dml()

    if config.use_cuda:
        assert torch.cuda.is_available(), "CUDA requested but not available"
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()

        for i in range(n_gpus):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f}GB)")

        scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        if scaler:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  AMP + cuDNN benchmark enabled")
    else:
        try:
            from usaf.utils import get_dml_device
            import torch_directml_native
            torch_directml_native.disable_tiled_resources(True)
            device = get_dml_device()
        except ImportError:
            device = torch.device("cpu")
        n_gpus = 1
        scaler = None

    return device, n_gpus, scaler


def main(args=None):
    config = parse_args(args)
    
    print("=" * 60)
    print("USAF â€” Ultra Sparse Adaptive Fine-Tuning")
    print("=" * 60)
    
    print(f"\nModel: {config.model_path}")
    from usaf.model_factory import detect_model, get_trainable_layers, get_param_patterns, get_router_path
    
    vram = torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0
    moe_cfg = detect_model(config.model_path, vram_gb=vram)
    
    if not moe_cfg.is_moe:
        print("Error: Model is not a Mixture-of-Experts architecture.")
        print("USAF only supports MoE models (Qwen3-MoE, Mixtral, DeepSeek-MoE, OLMoE, etc.)")
        sys.exit(1)
    
    print(f"Architecture: {moe_cfg.num_layers} layers, H={moe_cfg.hidden_size}, "
          f"heads={moe_cfg.num_attention_heads}/{moe_cfg.num_key_value_heads}")
    print(f"MoE: {moe_cfg.num_experts} experts, {moe_cfg.num_experts_per_tok} active, "
          f"intermediate={moe_cfg.expert_intermediate}")
    
    train_layers = get_trainable_layers(moe_cfg, config.train_from)
    print(f"Trainable: {len(train_layers)} layers "
          f"({min(train_layers) if train_layers else 0}-{max(train_layers) if train_layers else 0})")
    print(f"Param naming: {moe_cfg.expert_prefix} -> {moe_cfg.expert_param_names}")
    print(f"Router: {moe_cfg.router_path}")
    
    print(f"\nBackend: {'CUDA' if config.use_cuda else 'DirectML/CPU'}")
    device, n_gpus, scaler = setup_device(config)
    
    if config.use_multi_gpu and n_gpus > 1 and config.use_cuda:
        print(f"Multi-GPU: DataParallel across {n_gpus} GPUs")
    
    print(f"\nDataset: {config.dataset_path}")
    train_samples, eval_samples, heldout_samples = _load_dataset(
        config.dataset_path, config.seq_len)
    
    if config.epochs > 0:
        eff_batch = config.microbatch * config.accum
        tokens_per_epoch = len(train_samples) * config.seq_len
        config.steps = max(1, int(config.epochs * tokens_per_epoch / (eff_batch * config.seq_len)))
    
    eff_batch = config.microbatch * config.accum
    print(f"Steps: {config.steps}, Batch: {config.microbatch}Ã—{config.accum}={eff_batch}")
    print(f"Tokens: {config.steps * eff_batch * config.seq_len:,}")
    
    if not config.quant_path:
        model_name = config.model_path.split("/")[-1]
        config.quant_path = f"{model_name}-q4/experts_q4.pt"
    print(f"Q4 weights: {config.quant_path}")
    
    print(f"\nLoading model...")
    model, cache, q_dict, wf, st_path = _load_model(config, moe_cfg, device)
    
    if config.use_multi_gpu and n_gpus > 1 and config.use_cuda:
        model = nn.DataParallel(model)
    
    param_patterns = get_param_patterns(moe_cfg)
    _train_names = []
    for li in sorted(train_layers):
        _train_names.extend(param_patterns[li])
    
    def _q_shape(fn):
        e = q_dict[fn]
        if isinstance(e, dict):
            return tuple(e["shape"])
        return tuple(e[3])
    
    _shapes = {fn: _q_shape(fn) for fn in _train_names}
    
    base = model.module if hasattr(model, 'module') else model
    if hasattr(base, 'model') and hasattr(base.model, 'layers'):
        transformer = base.model
    elif hasattr(base, 'transformer') and hasattr(base.transformer, 'layers'):
        transformer = base.transformer
    else:
        raise RuntimeError("Cannot find transformer layers in model. Expected .model.layers or .transformer.layers")
    
    layers = transformer.layers
    embed = transformer.embed_tokens
    rotary = transformer.rotary_emb
    norm_fn = transformer.norm
    lm_head = base.lm_head
    
    resume_ckpt = None
    if config.resume_path and os.path.exists(config.resume_path):
        print(f"Resuming from checkpoint: {config.resume_path}")
        from usaf.checkpoint import load_sparse_checkpoint
        resume_ckpt = load_sparse_checkpoint(config.resume_path)
        print(f"  Resumed at step {resume_ckpt.get('step', 0)}, "
              f"{len(resume_ckpt.get('losses', []))} logged losses")

    print(f"\nStarting training...")
    print(f"  Sparsity: {config.frac*100:.1f}%")
    print(f"  RigL: every {config.reselect_every} steps")
    print(f"  Resident: {config.use_resident}")
    print(f"  Frozen cache: {config.use_frozen_cache}")
    if resume_ckpt:
        print(f"  Resume: step {resume_ckpt['step']}")
    if config.export_path:
        print(f"  Export: {config.export_path}")
    print(f"  RAM: {ram():.1f}GB\n")
    
    if config.eval_only:
        print("\n=== Eval-only mode (skipping training) ===\n")
        from usaf.eval.benchmark import run_benchmark, BenchmarkConfig
        from usaf.eval.report import save_report
        
        ds_list = [d.strip() for d in config.eval_datasets.split(",") if d.strip()]
        eval_cfg = BenchmarkConfig(
            datasets=ds_list,
            max_samples=config.eval_samples,
            seq_len=config.seq_len,
        )
        results = run_benchmark(
            model, model.tokenizer if hasattr(model, "tokenizer") else _get_tokenizer(),
            device, eval_cfg,
            model_name=config.model_path,
        )
        results.print()
        if config.eval_report:
            save_report(results, config.eval_report)
        return model
    
    _run_training(config, moe_cfg, model, cache, q_dict, device, scaler,
                  train_samples, eval_samples, heldout_samples,
                  _train_names, _shapes, train_layers,
                  layers, embed, rotary, norm_fn, lm_head,
                  resume_ckpt=resume_ckpt)
    
    return model


def _load_dataset(path: str, seq_len: int):
    """Load a JSONL dataset of tokenized sequences."""
    import random as _random
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
    
    _random.seed(42)
    _random.shuffle(samples)
    
    n_train = max(1, len(samples) - 20)
    return samples[:n_train], samples[n_train:n_train+10], samples[n_train+10:n_train+20]


def _load_model(config: TrainConfig, moe_cfg, device: torch.device):
    """Load model with quantized expert streaming."""
    from transformers import AutoConfig
    from safetensors import safe_open
    
    with torch.device("meta"):
        cfg = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
        from transformers import AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        except Exception:
            from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
            model = Qwen3MoeForCausalLM(cfg)
    
    import glob as _glob
    if os.path.isdir(config.model_path):
        st_path = config.model_path
    else:
        from transformers.utils import cached_file
        st_path = str(Path(cached_file(config.model_path, "config.json")).parent)
    
    st_files = sorted(_glob.glob(os.path.join(st_path, "*.safetensors")))
    wf = {}
    for fn in st_files:
        with safe_open(fn, framework="pt") as sf:
            for key in sf.keys():
                wf[key] = os.path.basename(fn)
    
    mp = dict(model.named_parameters())
    n_loaded = 0
    for name in sorted(wf.keys()):
        if ".mlp.experts." in name or ".block_sparse_moe." in name:
            continue
        if name not in mp:
            continue
        with safe_open(os.path.join(st_path, wf[name]), framework="pt") as sf:
            tensor = sf.get_tensor(name).half()
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        obj._parameters[parts[-1]] = nn.Parameter(
            tensor.to(device=device), requires_grad=False)
        n_loaded += 1
    
    for mn, mod in model.named_modules():
        for bn, b in list(mod._buffers.items()):
            if b is not None and b.device.type == "meta":
                if bn == "inv_freq":
                    hd_v = getattr(mod, "dim", getattr(mod, "head_dim", 128))
                    base = getattr(mod, "base", 1000000.0)
                    inv = 1.0 / (base ** (torch.arange(0, hd_v, 2, dtype=torch.float32) / hd_v))
                    mod._buffers[bn] = inv.to(dtype=torch.float16, device=device)
    
    print(f"  {n_loaded} non-expert params loaded")
    
    q_dict = torch.load(config.quant_path, map_location="cpu", weights_only=True)
    from usaf.moe_loader import QuantizedExpertCache
    cache = QuantizedExpertCache(q_dict, device, max_cached=1, group_size=128)
    
    for mname, mod in model.named_modules():
        if not (mname.endswith(".mlp.experts") or mname.endswith(".block_sparse_moe.experts")):
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
    
    return model, cache, q_dict, wf, st_path


def _run_training(config, moe_cfg, model, cache, q_dict, device, scaler,
                  train_samples, eval_samples, heldout_samples,
                  _train_names, _shapes, train_layers,
                  layers, embed, rotary, norm_fn, lm_head,
                  resume_ckpt=None):
    """Run the full training loop using pre-extracted layer references."""
    
    N_LAYERS = moe_cfg.num_layers
    DETACH_AT = min(train_layers) - 1
    MICROBATCH = config.microbatch
    ACCUM = config.accum
    SEQ = config.seq_len
    FRAC = config.frac
    LR_PEAK = config.lr_peak
    WD = config.weight_decay
    STEPS = config.steps
    RESELECT_EVERY = config.reselect_every
    USE_FROZEN_CACHE = config.use_frozen_cache
    USE_RESIDENT = config.use_resident
    
    from usaf.moe_loader import TopKImportanceStore, SparseGradStore
    from usaf.sparse_optim import SparseAdam
    from usaf.quantization import dequantize_4bit
    from usaf.checkpoint import save_sparse_checkpoint
    
    def _prelude(input_ids):
        hidden = embed(input_ids)
        s_len = hidden.shape[1]
        pos_ids = torch.arange(s_len, device=device).unsqueeze(0)
        cos, sin = rotary(hidden, position_ids=pos_ids)
        mask = torch.triu(
            torch.full((s_len, s_len), torch.finfo(torch.float16).min, device=device, dtype=torch.float16),
            diagonal=1).unsqueeze(0).unsqueeze(0)
        return hidden, pos_ids, (cos, sin), mask
    
    def _head_loss(hidden, labels):
        h = norm_fn(hidden)
        logits = lm_head(h)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    if resume_ckpt is None:
        imp_store = TopKImportanceStore(_shapes, frac=FRAC)
        for mname, mod in model.named_modules():
            if not (mname.endswith(".mlp.experts") or mname.endswith(".block_sparse_moe.experts")):
                continue
            mod._grad_capture = (imp_store, mname)
        
        def fwd_bwd_imp(sample):
            ids = torch.tensor([sample["input_ids"]], dtype=torch.long).to(device)
            lbl = torch.tensor([sample["labels"]], dtype=torch.long).to(device)
            hidden, pos_ids, pe, mask = _prelude(ids)
            for i in range(N_LAYERS):
                hidden = layers[i](hidden, attention_mask=mask, position_ids=pos_ids, position_embeddings=pe)
            cache.evict_all()
            h_last = hidden.detach().requires_grad_(True)
            loss = _head_loss(h_last, lbl)
            loss.backward()
            return loss.item()
        
        print("Importance phase...")
        t0 = time.time()
        N_IMP = 3 if not os.environ.get("SMOKE_N") else 1
        for imp_i in range(N_IMP):
            s = train_samples[imp_i % len(train_samples)]
            loss_imp = fwd_bwd_imp(s)
            print(f"  imp {imp_i+1}/{N_IMP} | loss {loss_imp:.4f} | {time.time()-t0:.0f}s")
        
        active_idx = imp_store.select(FRAC)
        ta = sum(i.numel() for i in active_idx.values())
        te = sum(math.prod(_shapes[fn]) for fn in active_idx if fn in _shapes)
        print(f"Active: {ta:,}/{te:,} ({100*ta/max(te,1):.4f}%)")
        
        masters = {}
        for fname, aidx in active_idx.items():
            aidx = aidx.reshape(-1).to(torch.long)
            entry = q_dict.get(fname)
            if entry is None:
                continue
            if isinstance(entry, dict):
                t = dequantize_4bit(entry["q"], entry["s"], entry["z"], entry["shape"], group_size=128)
            else:
                t = dequantize_4bit(entry[0], entry[1], entry[2], entry[3], group_size=128)
            vals = t.reshape(-1).index_select(0, aidx).float()
            del t
            p = nn.Parameter(vals, requires_grad=False)
            masters[fname] = p
            cache.overlays[fname] = (aidx, p)
        
        losses = []
        start_step = 1
    else:
        print(f"Resuming: restoring active_idx ({len(resume_ckpt['active_idx'])} tensors) "
              f"and masters ({len(resume_ckpt['masters'])} tensors)")
        active_idx = {k: v.to(torch.long) for k, v in resume_ckpt["active_idx"].items()}
        masters = {}
        for fname, vals in resume_ckpt["masters"].items():
            p = nn.Parameter(vals.float(), requires_grad=False)
            masters[fname] = p
            aidx = active_idx[fname].reshape(-1).to(torch.long)
            cache.overlays[fname] = (aidx, p)
        losses = list(resume_ckpt.get("losses", []))
        start_step = resume_ckpt.get("step", 0) + 1
        if start_step > STEPS:
            print(f"Checkpoint step {start_step-1} >= total steps {STEPS}, nothing to train")
            return losses
    
    sparse_store = SparseGradStore(active_idx, _shapes)
    for mname, mod in model.named_modules():
        if not (mname.endswith(".mlp.experts") or mname.endswith(".block_sparse_moe.experts")):
            continue
        mod._grad_capture = (sparse_store, mname)
    
    if USE_RESIDENT and resume_ckpt is None:
        cache.make_resident(train_layers)
        cache.apply_resident_overlays(active_idx, masters)
        cache._prefetch_disabled = True
    
    opt = SparseAdam(masters, active_idx=active_idx, lr=LR_PEAK, weight_decay=WD, compact_params=True)
    if resume_ckpt is not None and "optimizer" in resume_ckpt:
        opt.load_state_dict(resume_ckpt["optimizer"])
        print(f"  Optimizer state restored (step {opt._step})")
    print(f"Optimizer: {opt.optimizer_memory_mb:.1f}MB")
    
    def fwd_bwd(batch, zero_store=True):
        if isinstance(batch, dict):
            batch = [batch]
        if zero_store:
            sparse_store.zero_()
        ids = torch.stack([torch.tensor(s["input_ids"], dtype=torch.long) for s in batch]).to(device)
        lbl = torch.stack([torch.tensor(s["labels"], dtype=torch.long) for s in batch]).to(device)
        hidden, pos_ids, pe, mask = _prelude(ids)
        
        with torch.no_grad():
            for i in range(DETACH_AT + 1):
                hidden = layers[i](hidden, attention_mask=mask, position_ids=pos_ids, position_embeddings=pe)
            cache.evict_all()
            xs = []
            for i in range(DETACH_AT + 1, N_LAYERS):
                xs.append(hidden)
                hidden = layers[i](hidden, attention_mask=mask, position_ids=pos_ids, position_embeddings=pe)
            cache.evict_all()
        
        h_last = hidden.detach().requires_grad_(True)
        loss = _head_loss(h_last, lbl)
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        g = h_last.grad
        for j in range(len(xs) - 1, -1, -1):
            i = DETACH_AT + 1 + j
            x2 = xs[j].detach().requires_grad_(True)
            out = layers[i](x2, attention_mask=mask, position_ids=pos_ids, position_embeddings=pe)
            out.backward(g)
            g = x2.grad
            cache.evict_all()
        
        return loss.item()
    
    losses = []
    si = 0
    t_start = time.time()
    good_streak = 0
    loss_scale = 4096.0
    
    print(f"\n=== Training ({STEPS} steps) ===\n")
    for step in range(start_step, STEPS + 1):
        t_step = time.time()
        
        if step <= max(1, int(STEPS * 0.05)):
            lr = LR_PEAK * step / max(1, int(STEPS * 0.05))
        else:
            progress = (step - max(1, int(STEPS * 0.05))) / max(1, STEPS - max(1, int(STEPS * 0.05)))
            lr = LR_PEAK * 0.1 + LR_PEAK * 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        opt.lr = lr
        
        sparse_store.zero_()
        step_loss = 0.0
        
        for a in range(ACCUM):
            mb = []
            for _ in range(MICROBATCH):
                mb.append(train_samples[si % len(train_samples)])
                si += 1
                if si % len(train_samples) == 0:
                    random.shuffle(train_samples)
            lv = fwd_bwd(mb, zero_store=False)
            step_loss += lv
        
        step_loss /= ACCUM
        
        denom = loss_scale * ACCUM
        cg = {n: v / denom for n, v in sparse_store.compact.items()}
        finite = all(torch.isfinite(v).all().item() for v in cg.values())
        
        if finite:
            opt.step(compact_grads=cg)
            if USE_RESIDENT:
                cache.sync_resident(active_idx, masters)
            good_streak += 1
            if good_streak % 200 == 0:
                loss_scale = min(loss_scale * 2, 65536.0)
        else:
            loss_scale = max(loss_scale / 2, 64.0)
        
        cache.evict_all()
        losses.append(step_loss)
        
        dt = time.time() - t_step
        pct = 100.0 * step / STEPS
        eta_h = (STEPS - step) * dt / 3600
        tok_s = ACCUM * MICROBATCH * SEQ / dt
        
        log_msg = f"  {step:3d}/{STEPS} | loss {step_loss:.4f} | {tok_s:.0f} tok/s | LR {lr:.1e} | RAM {ram():.1f}G | ETA {eta_h:.1f}h"
        if resume_ckpt is not None:
            log_msg += f" | resumed"
        print(log_msg, flush=True)
        
        if config.save_every > 0 and step % config.save_every == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"sparse_step-{step}.pt")
            save_sparse_checkpoint(
                ckpt_path, masters, active_idx, opt.state_dict(),
                {"model": config.model_path, "steps": STEPS, "frac": FRAC,
                 "lr": LR_PEAK, "seq_len": SEQ, "microbatch": MICROBATCH, "accum": ACCUM,
                 "train_from": config.train_from, "reselect_every": RESELECT_EVERY,
                 "save_every": config.save_every, "tag": config.tag},
                step, losses, list(train_layers), metric=step_loss,
            )
            print(f"  >>> checkpoint saved: {ckpt_path}", flush=True)
    
    t_total = time.time() - t_start
    skipped = sum(1 for l in losses if not math.isfinite(l))
    
    print(f"\n=== Complete ===")
    print(f"Time: {t_total/3600:.1f}h")
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"Skipped: {skipped}/{STEPS} steps")
    print(f"Peak RAM: {ram():.1f}GB")
    
    if config.export_path:
        print(f"\nExporting merged weights to {config.export_path}...")
        try:
            from usaf.checkpoint import export_merged_weights
            export_path = export_merged_weights(
                config.quant_path,
                {k: v.detach().cpu() for k, v in masters.items()},
                {k: v.cpu() for k, v in active_idx.items()},
                config.export_path,
            )
            print(f"  Exported: {export_path}")
        except Exception as e:
            print(f"  Export failed: {e}")
    
    return losses


if __name__ == "__main__":
    main()
