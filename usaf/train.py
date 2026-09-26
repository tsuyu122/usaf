"""USAF: Ultra Sparse Adaptive Fine-Tuning - Universal Training CLI.

Supports any MoE model from HuggingFace. Auto-detects architecture and configures training.

Usage:
    usaf train --model Qwen/Qwen3-30B-A3B --dataset data/train.jsonl
    usaf train --model deepseek-ai/DeepSeek-MoE-16B --dataset data.jsonl --steps 360
    python -m usaf.train --help
"""
import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch
import torch.nn as nn


def ram() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def _get_tokenizer(model_path: str | None = None):
    """Return a tokenizer for ``model_path``, or None if there is none.

    This used to return a global set by _set_tokenizer(), which in turn was
    only ever fed from ``model.tokenizer``. Models are built here with
    ``from_config`` and carry no tokenizer attribute at all, so the global
    was always None and --eval-only died with a bare
    "'NoneType' object is not callable" from deep inside perplexity.py.
    """
    tok = getattr(_get_tokenizer, "_instance", None)
    if tok is not None:
        return tok
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"  [warn] no tokenizer for {model_path}: {type(e).__name__}")
        return None
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    _set_tokenizer(tok)
    return tok


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
    use_cuda: bool | None = None
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


def setup_device(config: TrainConfig) -> tuple[torch.device, int, object]:
    """Configure device, AMP scaler, and multi-gPU."""
    # Install the dense-masked MoE forward patches. These are what route the
    # sparse expert gradients into the capture store: without them the native
    # forward never calls _grad_capture and sparse training silently does
    # nothing. Each family gets its own patch because the expert container
    # class and the router API differ, even though from transformers>=4.53
    # they all share the fused gate_up_proj/down_proj layout.
    from usaf.mixtral_dml import patch_mixtral_for_dml
    from usaf.olmoe_dml import patch_olmoe_for_dml
    from usaf.qwen3moe_dml import patch_qwen3moe_for_dml
    patch_qwen3moe_for_dml()
    patch_olmoe_for_dml()
    patch_mixtral_for_dml()

    if config.use_cuda:
        assert torch.cuda.is_available(), "CUDA requested but not available"
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()

        for i in range(n_gpus):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f}GB)")

        # NOTE: deliberately no torch.cuda.amp.GradScaler. The sparse-gradient
        # path captures grads into a custom store via module hooks, not into
        # param.grad, so GradScaler.unscale_/step/update would never run and its
        # fixed 65536 scale factor would leak into the captured gradients. Manual
        # loss scaling (loss*loss_scale before backward, divided by
        # loss_scale*ACCUM in the step loop) is used instead - the same scheme
        # the working DirectML path in the root train.py uses.
        scaler = None
        if config.use_amp:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  AMP (manual loss scaling) + cuDNN benchmark + TF32 enabled")
    else:
        try:
            import torch_directml_native

            from usaf.utils import get_dml_device
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
    print("USAF — Ultra Sparse Adaptive Fine-Tuning")
    print("=" * 60)

    print(f"\nModel: {config.model_path}")
    from usaf.model_factory import detect_model, get_param_patterns, get_trainable_layers

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

    config.quant_path = _resolve_quant_path(
        config.quant_path, config.model_path.split("/")[-1]
    )
    print(f"Q4 weights: {config.quant_path}")

    print("\nLoading model...")
    model, cache, q_dict, wf, st_path, router_params = _load_model(
        config, moe_cfg, device, train_layers)

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
    resume_path = _check_resume_path(config.resume_path)
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        from usaf.checkpoint import load_sparse_checkpoint
        resume_ckpt = load_sparse_checkpoint(resume_path)
        print(f"  Resumed at step {resume_ckpt.get('step', 0)}, "
              f"{len(resume_ckpt.get('losses', []))} logged losses")

    print("\nStarting training...")
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
        from usaf.eval.benchmark import BenchmarkConfig, run_benchmark
        from usaf.eval.report import save_report

        ds_list = [d.strip() for d in config.eval_datasets.split(",") if d.strip()]
        if not ds_list:
            raise SystemExit(
                "--eval-only needs at least one dataset in --eval-datasets"
            )
        tokenizer = _get_tokenizer(config.model_path)
        if tokenizer is None:
            raise SystemExit(
                f"--eval-only needs a tokenizer, and none could be loaded "
                f"from {config.model_path}. A model built with from_config "
                f"has none, so save one next to the weights "
                f"(tokenizer.json / tokenizer_config.json), or pass a model "
                f"id that ships one. Use the normal training path if you only "
                f"want perplexity on the training dataset."
            )
        # A tokenizer whose vocabulary does not fit the model produces token
        # ids past the embedding table, which used to crash deep inside
        # torch.embedding with a bare "index out of range in self".
        model_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        tok_vocab = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0) or 0)
        if model_vocab and tok_vocab > model_vocab:
            raise SystemExit(
                f"--eval-only: the tokenizer produces {tok_vocab} distinct "
                f"ids but {config.model_path} has an embedding table of only "
                f"{model_vocab}. These are not the same model's tokenizer."
            )
        eval_cfg = BenchmarkConfig(
            datasets=ds_list,
            max_samples=config.eval_samples,
            seq_len=config.seq_len,
        )
        results = run_benchmark(
            model, tokenizer, device, eval_cfg,
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
                  router_params=router_params,
                  resume_ckpt=resume_ckpt)

    return model


def _check_resume_path(resume_path: str) -> str:
    """Validate an explicit --resume path, or return None when unused.

    The path used to be guarded by os.path.exists, so a typo silently started a
    fresh run from step zero and exited 0. On a long run that is hours of GPU
    time discarded, and the log still read "Starting training..." as if
    nothing were wrong.
    """
    if not resume_path:
        return None
    if os.path.isdir(resume_path):
        raise SystemExit(
            f"--resume takes a checkpoint file, not a directory: {resume_path}"
        )
    if not os.path.exists(resume_path):
        raise SystemExit(
            f"checkpoint not found: {resume_path}\n"
            f"  Checkpoints are written to --checkpoint-dir as "
            f"sparse_step-N.pt. Refusing to start from scratch."
        )
    return resume_path


def _resolve_quant_path(quant_path: str, model_name: str) -> str:
    """Work out which file holds the quantized expert weights.

    --quant-path names a file, but a directory is the natural thing to pass and
    torch.load reports a directory as "PermissionError: [Errno 13] Permission
    denied", which reads like a lock rather than a wrong path. Accept the
    directory form, and say exactly what is expected when nothing is there.
    """
    if not quant_path:
        quant_path = f"{model_name}-q4/experts_q4.pt"
    if os.path.isdir(quant_path):
        quant_path = os.path.join(quant_path, "experts_q4.pt")
    if not os.path.exists(quant_path):
        raise SystemExit(
            f"quantized weights not found: {quant_path}\n"
            f"  --quant-path takes the experts_q4.pt file, or the directory "
            f"containing it. Quantize the model first."
        )
    return quant_path


def _load_dataset(path: str, seq_len: int):
    """Load a JSONL dataset of tokenized sequences.

    Every way this can go wrong used to surface somewhere else entirely: a
    missing file was treated the same as an empty one, malformed lines were
    dropped without a word, and rows whose input_ids was not a list raised
    "TypeError: object of type str has no len()". An empty result then produced
    "ZeroDivisionError: integer modulo by zero" from the progress bar several
    hundred lines later. Each failure is now reported where it happens.
    """
    import random as _random

    if not os.path.exists(path):
        raise SystemExit(f"dataset not found: {path}")

    samples = []
    bad_json = 0
    wrong_len = 0
    malformed = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                if bad_json <= 3:
                    print(f"  {path}:{lineno}: not valid JSON, skipping")
                continue
            ids = s.get("input_ids") if isinstance(s, dict) else None
            if not isinstance(ids, list):
                malformed += 1
                if malformed <= 3:
                    print(f"  {path}:{lineno}: input_ids is not a list, skipping")
                continue
            if len(ids) != seq_len:
                wrong_len += 1
                continue
            samples.append(s)

    if bad_json > 3:
        print(f"  {path}: {bad_json} unparseable lines in total")
    if malformed > 3:
        print(f"  {path}: {malformed} lines without a list of input_ids in total")

    if not samples:
        raise SystemExit(
            f"no usable sample in {path}: {len(samples)} of the lines are exactly "
            f"{seq_len} token ids. Fix --seq-len, or regenerate the dataset."
        )
    if wrong_len:
        print(f"  {wrong_len} line(s) skipped: not exactly {seq_len} tokens")

    _random.seed(42)
    _random.shuffle(samples)

    n_train = max(1, len(samples) - 20)
    return samples[:n_train], samples[n_train:n_train+10], samples[n_train+10:n_train+20]


def _expert_module_names(moe_cfg) -> set[str]:
    """Return the set of expert module names for this architecture.

    Derived from the *detected* expert_prefix (e.g. "model.layers.{i}.mlp.experts")
    instead of hardcoding a ".mlp.experts" suffix. The previous suffix check
    never matched architectures whose experts live under a different path,
    silently leaving the forward hooks uninstalled.
    """
    return {moe_cfg.expert_prefix.format(i=i) for i in range(moe_cfg.num_layers)}


def _rebuild_inv_freq(rotary_mod, cfg) -> torch.Tensor:
    """Recompute a RoPE module inv_freq buffer from the model config.

    The model is materialised on the meta device, so every buffer - including
    inv_freq - starts out without real storage. We rebuild it with the same
    routine transformers itself uses (ROPE_INIT_FUNCTIONS), driven by the
    config, so the rotary dimension, theta and any rope_scaling are honoured
    instead of being guessed.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    # rope_type lives in config.rope_parameters on transformers 5.x and in
    # config.rope_scaling (or top-level rope_theta) on 4.x. The "default"
    # type has no entry in ROPE_INIT_FUNCTIONS - it is computed by the rotary
    # class itself, so we dispatch to that static method when present.
    rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    rope_type = rp.get("rope_type", rp.get("type", "default"))
    default_fn = getattr(type(rotary_mod), "compute_default_rope_parameters", None)
    if rope_type == "default" and callable(default_fn):
        # transformers 5.17 deprecates the device kwarg (removed in 5.18).
        # Inspect the signature so we neither emit a FutureWarning on 5.17
        # nor crash with a TypeError on 5.18+.
        import inspect as _inspect
        _params = _inspect.signature(default_fn).parameters
        if "device" in _params:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                inv_freq, attn_scaling = default_fn(cfg, device=torch.device("cpu"))
        else:
            inv_freq, attn_scaling = default_fn(cfg)
    else:
        if rope_type not in ROPE_INIT_FUNCTIONS:
            raise ValueError(f"unsupported rope_type {rope_type!r}")
        inv_freq, attn_scaling = ROPE_INIT_FUNCTIONS[rope_type](
            cfg, device=torch.device("cpu"), seq_len=None
        )
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    # partial_rotary_factor shrinks the rotary dim below head_dim (Phi-3 etc).
    prf = getattr(cfg, "partial_rotary_factor", 1.0) or 1.0
    rot_dim = int(head_dim * prf)
    # Some rope init helpers return a bare tensor rather than a
    # (inv_freq, scaling) tuple depending on version; normalise both.
    if isinstance(inv_freq, (tuple, list)):
        inv_freq, attn_scaling = inv_freq[0], inv_freq[1]
    if attn_scaling and attn_scaling != 1.0 and hasattr(rotary_mod, "attention_scaling"):
        rotary_mod.attention_scaling = attn_scaling
    expected = rot_dim // 2
    if inv_freq.shape[0] != expected:
        raise ValueError(
            f"rebuilt RoPE inv_freq has {inv_freq.shape[0]} entries but "
            f"rotary dim {rot_dim} (head_dim={head_dim}, "
            f"partial_rotary_factor={prf}) requires {expected}"
        )
    return inv_freq


def _layer_is_trainable(param_name: str, train_layers: set[int]) -> bool:
    """Return True if param_name belongs to one of the trainable layers.

    Parses "model.layers.<i>...." and compares <i> against the trainable set.
    """
    parts = param_name.split(".")
    if "layers" not in parts:
        return False
    try:
        idx = int(parts[parts.index("layers") + 1])
    except (IndexError, ValueError):
        return False
    return idx in train_layers


def _load_model(config: TrainConfig, moe_cfg, device: torch.device,
                train_layers: set[int]):
    """Load model with quantized expert streaming."""
    from safetensors import safe_open
    from transformers import AutoConfig

    with torch.device("meta"):
        cfg = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
        from transformers import AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        except Exception:
            from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
            model = Qwen3MoeForCausalLM(cfg)

    # Router bookkeeping: suffix (from the detected config) and the dict of
    # trainable gate params, returned so _run_training can optimize them.
    router_suffix = moe_cfg.router_path
    router_params: dict[str, nn.Parameter] = {}

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

    _expert_modules = _expert_module_names(moe_cfg)
    mp = dict(model.named_parameters())
    n_loaded = 0
    for name in sorted(wf.keys()):
        if any(name.startswith(m + ".") for m in _expert_modules):
            continue
        if name not in mp:
            continue
        with safe_open(os.path.join(st_path, wf[name]), framework="pt") as sf:
            tensor = sf.get_tensor(name).half()
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        # The router (gate) of each trainable layer is trained alongside the
        # sparse experts. For MoE models the gate decides which experts fire
        # and is the single highest-leverage parameter to update - LoRA-style
        # adapter methods cannot touch it. Mark it requires_grad and collect
        # it for a separate optimizer (see _run_training).
        train_gate = (name.endswith(router_suffix) and _layer_is_trainable(name, train_layers))
        obj._parameters[parts[-1]] = nn.Parameter(
            tensor.to(device=device), requires_grad=train_gate)
        if train_gate:
            router_params[name] = obj._parameters[parts[-1]]
        n_loaded += 1

    for mn, mod in model.named_modules():
        for bn, b in list(mod._buffers.items()):
            if b is not None and b.device.type == "meta":
                if bn == "inv_freq":
                    # Rebuild RoPE exactly the way transformers does, from
                    # the config. The previous code guessed the rotary dim
                    # with getattr(mod, "dim", getattr(mod, "head_dim", 128)),
                    # but rotary modules expose neither attribute, so it
                    # always fell through to a hardcoded 128. On a model
                    # whose real head_dim differs that produced an inv_freq
                    # of the wrong length, and the forward died with
                    # "The size of tensor a (8) must match tensor b (128)".
                    mod._buffers[bn] = _rebuild_inv_freq(mod, cfg).to(
                        device=device)

    print(f"  {n_loaded} non-expert params loaded")

    q_dict = torch.load(config.quant_path, map_location="cpu", weights_only=True)
    from usaf.moe_loader import QuantizedExpertCache
    cache = QuantizedExpertCache(q_dict, device, max_cached=1, group_size=128,
                                expert_prefix=moe_cfg.expert_prefix)

    for mname, mod in model.named_modules():
        if mname not in _expert_modules:
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

    return model, cache, q_dict, wf, st_path, router_params


def _run_training(config, moe_cfg, model, cache, q_dict, device, scaler,
                  train_samples, eval_samples, heldout_samples,
                  _train_names, _shapes, train_layers,
                  layers, embed, rotary, norm_fn, lm_head,
                  router_params=None,
                  resume_ckpt=None):
    """Run the full training loop using pre-extracted layer references."""

    N_LAYERS = moe_cfg.num_layers
    _expert_modules = _expert_module_names(moe_cfg)
    DETACH_AT = min(train_layers) - 1
    MICROBATCH = config.microbatch
    ACCUM = config.accum
    SEQ = config.seq_len
    FRAC = config.frac
    LR_PEAK = config.lr_peak
    WD = config.weight_decay
    STEPS = config.steps
    RESELECT_EVERY = config.reselect_every
    USE_RESIDENT = config.use_resident

    from usaf.checkpoint import save_sparse_checkpoint
    from usaf.moe_loader import SparseGradStore, TopKImportanceStore
    from usaf.quantization import dequantize_4bit
    from usaf.sparse_optim import SparseAdam

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
            if mname not in _expert_modules:
                continue
            mod._grad_capture = (imp_store, mname)

        def fwd_bwd_imp(sample):
            """Forward+backward used only by the importance phase.

            The expert weights are injected into the modules by forward
            pre-hooks, so the sparse-gradient hooks registered on them only
            fire if the backward pass actually walks back through the decoder
            layers. Detaching the final hidden state (h_last) severs the graph
            produced by the forward loop, so a plain loss.backward() would only
            ever reach h_last itself and the experts would receive no gradient
            at all.

            We therefore keep the per-layer inputs, and after computing the
            head loss we re-run each captured layer in reverse, threading the
            incoming gradient through it manually. This mirrors what
            fwd_bwd() below does during real training.
            """
            ids = torch.tensor([sample["input_ids"]], dtype=torch.long).to(device)
            lbl = torch.tensor([sample["labels"]], dtype=torch.long).to(device)
            hidden, pos_ids, pe, mask = _prelude(ids)
            xs_imp = []
            for i in range(N_LAYERS):
                xs_imp.append(hidden)
                hidden = layers[i](hidden, attention_mask=mask, position_ids=pos_ids, position_embeddings=pe)
            cache.evict_all()
            h_last = hidden.detach().requires_grad_(True)
            loss = _head_loss(h_last, lbl)
            loss.backward()
            g_imp = h_last.grad
            for j in range(len(xs_imp) - 1, -1, -1):
                x2 = xs_imp[j].detach().requires_grad_(True)
                out = layers[j](x2, attention_mask=mask, position_ids=pos_ids, position_embeddings=pe)
                out.backward(g_imp)
                g_imp = x2.grad
                cache.evict_all()
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
        if mname not in _expert_modules:
            continue
        mod._grad_capture = (sparse_store, mname)

    if USE_RESIDENT and resume_ckpt is None:
        cache.make_resident(train_layers)
        cache.apply_resident_overlays(active_idx, masters)
        cache._prefetch_disabled = True

    # Separate optimizer for the router (gate) weights.
    # Adam on fp16 DirectML falls back to CPU and produces NaN, so we use
    # plain SGD with momentum - the same choice the root train.py makes.
    router_opt = None
    if router_params:
        router_opt = torch.optim.SGD(list(router_params.values()), lr=LR_PEAK,
                                      momentum=0.9, weight_decay=WD)
        _rn = sum(p.numel() for p in router_params.values())
        print(f"  Router gates: {len(router_params)} params, {_rn:,} elements, SGD+momentum")

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

        # Scale the loss before backward so fp16 gradients do not underflow to
        # zero; the step loop divides the captured grads by loss_scale*ACCUM to
        # recover the true gradient. Reading loss_scale from the enclosing
        # scope is safe: fwd_bwd is only invoked from the training loop, after
        # loss_scale has been initialised.
        (loss * loss_scale).backward()

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
        if router_params:
            for p in router_params.values():
                p.grad = None
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
        # Router grads were captured by autograd in the re-backward pass and
        # are already divided by loss_scale there, so unscale by the
        # accumulation factor only (same treatment as the root train.py).
        for p in router_params.values() if router_params else ():
            if p.grad is not None:
                p.grad.data.div_(ACCUM)
        finite = all(torch.isfinite(v).all().item() for v in cg.values())
        if router_params:
            finite = finite and all(
                torch.isfinite(p.grad).all().item()
                for p in router_params.values() if p.grad is not None
            )

        if finite:
            opt.step(compact_grads=cg)
            if router_opt is not None:
                router_opt.step()
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
        100.0 * step / STEPS
        eta_h = (STEPS - step) * dt / 3600
        tok_s = ACCUM * MICROBATCH * SEQ / dt

        log_msg = f"  {step:3d}/{STEPS} | loss {step_loss:.4f} | {tok_s:.0f} tok/s | LR {lr:.1e} | RAM {ram():.1f}G | ETA {eta_h:.1f}h"
        if resume_ckpt is not None:
            log_msg += " | resumed"
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

    print("\n=== Complete ===")
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
