"""USAF: Ultra Sparse Adaptive Fine-Tuning - Universal Training CLI.

Supports any MoE model from HuggingFace. Auto-detects architecture and configures training.

Usage:
    usaf train --model Qwen/Qwen3-30B-A3B --dataset data/train.jsonl
    usaf train --model deepseek-ai/DeepSeek-MoE-16B --dataset data.jsonl --steps 360
    python -m usaf.train --help
"""
import argparse
import gc
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

from usaf.utils import resolve_dtype


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
    p.add_argument("--generate-after", type=str, default="",
                   help="Generate from this prompt after training ("" to skip); use || to separate several")
    p.add_argument("--generate-tokens", type=int, default=48)

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
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "fp16", "bf16", "fp32"],
                   help="Weight and activation precision. "
                        "auto picks bf16 on CUDA hardware that has it, "
                        "fp16 on older CUDA and DirectML, fp32 on CPU")

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
    generate_after: str = ""
    generate_tokens: int = 48
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
    dtype: str = "auto"


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
        generate_after=ns.generate_after,
        generate_tokens=ns.generate_tokens,
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
        dtype=ns.dtype,
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
    from usaf.zaya_dml import patch_zaya_for_dml

    patch_qwen3moe_for_dml()
    patch_olmoe_for_dml()
    patch_mixtral_for_dml()
    # ZAYA defines its own ZayaExperts, with no inheritance from Qwen3MoeExperts,
    # so the three patches above do not reach it. Without this the sparse capture
    # never fires, select() returns nothing, and the run trains only the
    # non-expert parameters with the loss falling throughout.
    patch_zaya_for_dml()

    if config.use_cuda:
        # An assert, not a check: python -O strips asserts entirely, so under -O
        # this silently fell through to building a CUDA device on a machine
        # with no CUDA, and the failure arrived later as a shape error inside
        # the model. It also printed "Backend: CUDA" before dying, so the last
        # thing on screen claimed a backend that was never used.
        if not torch.cuda.is_available():
            raise SystemExit(
                "--cuda was requested but torch reports no CUDA device.\n"
                f"  torch {torch.__version__} sees {torch.cuda.device_count()} GPUs.",
                "  Drop --cuda to use DirectML or the CPU."
            )
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
        # DirectML is tried first on a machine with no CUDA, which is the
        # intended fallback order. What is not acceptable is importing it
        # unguarded: a torch/directml mismatch fails inside the Windows
        # loader, and under faulthandler - which pytest turns on - that kills
        # the interpreter with STATUS_ENTRYPOINT_NOT_FOUND and no Python
        # exception, so `except ImportError` below never got to run and the
        # whole suite died here. has_dml() imports through the guarded loader
        # in usaf.utils, so an unusable install is known before anything native
        # is touched. Exception rather than ImportError for the same reason:
        # a backend that cannot start should degrade to CPU, not end the run.
        device = torch.device("cpu")
        try:
            from usaf.utils import get_dml_device, has_dml

            if has_dml():
                import torch_directml_native

                torch_directml_native.disable_tiled_resources(True)
                device = get_dml_device()
        except Exception:  # noqa: BLE001 - never fatal, fall back to CPU
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

    # setup_device first, then announce. Announcing first meant a run that died
    # a line later still left "Backend: CUDA" as the last thing on screen,
    # claiming a backend it never got to use.
    #
    # And the label is read off the device that came back, not off the flag
    # that was asked for. Deriving it from config.use_cuda claimed DirectML on
    # every non-CUDA run, including the ones where importing DirectML failed
    # and the device was the CPU - the same overclaim, one level down, and the
    # reason a run on the wrong backend is not obvious from the log. When
    # DirectML was asked for and is not there, that is said outright rather than
    # blended into a slash-separated label nobody can read as either thing.
    device, n_gpus, scaler = setup_device(config)
    _t = device.type
    if _t == "cuda":
        _backend = "CUDA"
    elif _t == "privateuseone":
        _backend = "DirectML"
    else:
        _backend = _t.upper()
    if not config.use_cuda and _backend == "CPU":
        print("\nBackend: CPU (DirectML was tried and is not available here)")
    else:
        print(f"\nBackend: {_backend}")

    if config.use_multi_gpu and n_gpus > 1 and config.use_cuda:
        # Do not claim a split that cannot happen. The forward loop below walks
        # layers[i](...) on submodules taken off model.module, never model(...),
        # and DataParallel only scatters when its own forward is called. The
        # wrapper is therefore never entered: every microbatch runs entirely on
        # device 0 and the second card sits idle. Two GPUs reported as
        # DataParallel and running on one is the worst kind of wrong here - it
        # costs nothing to notice and it makes a 2x machine look like a 1x one
        # in every timing.
        #
        # The sparse path also rules out a quick fix: experts are injected by
        # per-layer pre-hooks and the backward is threaded one layer at a time,
        # so there is no single call for DataParallel to scatter. Making this
        # real means sharding layers across devices, which is a redesign rather
        # than a flag.
        print(f"GPUs visible: {n_gpus} - forward on device 0 only, "
              f"the sparse path bypasses DataParallel")
    else:
        print("GPUs used: 1")

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

    # The whole model path, not its last segment.
    #
    # split on forward slash strips a directory on Linux and does nothing at
    # all on Windows, where the separator is a backslash. So the full path
    # reached the resolver here, the bare name reached it on Kaggle, and the
    # difference only ever showed up on the platform nobody develops on.
    # Resolving beside the model directory is the contract, and handing the
    # function a basename throws away exactly what it needs to do that.
    config.quant_path = _resolve_quant_path(
        config.quant_path, config.model_path
    )
    print(f"Q4 weights: {config.quant_path}")

    # A layer budget is an estimate, and the estimate for a model only ever
    # met on real hardware was optimistic: it chose 14 trainable layers for
    # ZAYA1-8B on a 15.6 GB T4, the run got 65 minutes in, and it died trying to
    # allocate 256 MB with 194 MB free. Rather than treat that as a lost run,
    # the load and the training loop are repeated with fewer layers: each OOM
    # drops the layer count to three quarters and starts again. The frozen
    # activation cache is rebuilt from scratch on the way in, so nothing from
    # the failed attempt is carried over.
    #
    # A shrink that cannot save memory never will, so at one layer the loop stops
    # and reports rather than grinding.
    def _memory_report(dev):
        """What the device held when it ran out, in one line.

        An out-of-memory retry only helps when the trainable layers were what
        filled the card. When they are not, the retry shrinks the model, fails
        again for exactly the same reason, and the run ends reporting that no
        memory error explained it while every single attempt was one. These
        numbers are the difference between those two cases being told apart.
        """
        try:
            if dev.type == "cuda":
                free_b, total_b = torch.cuda.mem_get_info(dev)
                used = (total_b - free_b) / 1e9
                peak = torch.cuda.max_memory_allocated(dev) / 1e9
                held = torch.cuda.max_memory_reserved(dev) / 1e9
                return (
                    f"; card {used:.2f} GB of {total_b / 1e9:.2f} in use, "
                    f"peak allocated {peak:.2f} GB, peak reserved {held:.2f} GB"
                )
            if dev.type == "cpu":
                return "; host RAM in use"
        except Exception:
            return ""
        return ""

    def _load_and_run(_layers):
        print(f"\nLoading model... (trainable: {len(_layers)} layers)")
        _model, _cache, _q_dict, _wf, _st, _router, _model_cfg = _load_model(
            config, moe_cfg, device, _layers)

        if config.use_multi_gpu and n_gpus > 1 and config.use_cuda:
            _model = nn.DataParallel(_model)

        _patterns = get_param_patterns(moe_cfg)
        _train_names = []
        for _li in sorted(_layers):
            _train_names.extend(_patterns[_li])

        def _q_shape(fn):
            e = _q_dict[fn]
            if isinstance(e, dict):
                return tuple(e["shape"])
            return tuple(e[3])

        _shapes = {fn: _q_shape(fn) for fn in _train_names}

        _base = _model.module if hasattr(_model, 'module') else _model
        if hasattr(_base, 'model') and hasattr(_base.model, 'layers'):
            _transformer = _base.model
        elif hasattr(_base, 'transformer') and hasattr(_base.transformer, 'layers'):
            _transformer = _base.transformer
        else:
            raise RuntimeError(
                "Cannot find transformer layers in model. Expected .model.layers"
                " or .transformer.layers")

        _layers_mod = _transformer.layers
        _embed = _transformer.embed_tokens
        _rotary = _transformer.rotary_emb
        _norm_fn = _transformer.norm
        _lm_head = _base.lm_head

        _resume = None
        _rpath = _check_resume_path(config.resume_path)
        if _rpath:
            print(f"Resuming from checkpoint: {_rpath}")
            from usaf.checkpoint import load_sparse_checkpoint
            _resume = load_sparse_checkpoint(_rpath)
            print(f"  Resumed at step {_resume.get('step', 0)}, "
                  f"{len(_resume.get('losses', []))} logged losses")

        print("\nStarting training...")
        print(f"  Sparsity: {config.frac*100:.1f}%")
        print(f"  RigL: every {config.reselect_every} steps")
        _eval_cfg = None
        if config.eval_only:
            print("\n=== Eval-only mode (skipping training) ===\n")
            from usaf.eval.benchmark import BenchmarkConfig, run_benchmark
            from usaf.eval.report import save_report

            _ds_list = [d.strip()
                        for d in config.eval_datasets.split(",") if d.strip()]
            tokenizer = _get_tokenizer(config.model_path)
            # torch.embedding with a bare "index out of range in self".
            _mv = getattr(getattr(_model, "config", None), "vocab_size", None)
            _tv = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0) or 0)
            if _mv and _tv > _mv:
                raise SystemExit(
                    f"--eval-only: the tokenizer produces {_tv} distinct "
                    f"ids but {config.model_path} has an embedding table of only "
                    f"{_mv}. These are not the same model's tokenizer."
                )
            _eval_cfg = BenchmarkConfig(
                datasets=_ds_list,
                max_samples=config.eval_samples,
                seq_len=config.seq_len,
            )
            if _resume is not None:
                _apply_resume_overlays(_resume, _cache)
                print(f"  Applied trained weights from {_resume.get('step', '?')} "
                      f"steps ({len(_resume['masters'])} tensors)")
            results = run_benchmark(
                _model, tokenizer, device, _eval_cfg,
                model_name=config.model_path,
            )
            results.print()
            if config.eval_report:
                save_report(results, config.eval_report)
            return _model

        _run_training(config, moe_cfg, _model, _cache, _q_dict, device, scaler,
                      train_samples, eval_samples, heldout_samples,
                      _train_names, _shapes, _layers,
                      _layers_mod, _embed, _rotary, _norm_fn, _lm_head,
                      router_params=_router,
                      model_cfg=_model_cfg,
                      resume_ckpt=_resume)
        return _model

    _layers_now = list(train_layers)
    model = None
    _attempt_sizes: list[int] = []
    _why = ""
    for _attempt in range(1, 6):
        _attempt_sizes.append(len(_layers_now))
        try:
            model = _load_and_run(_layers_now)
            break
        except torch.OutOfMemoryError as _oom:
            _why = str(_oom).splitlines()[0] if _oom.args else ""
            # What the card actually held when it gave up. Retrying with fewer
            # layers only helps if the trainable layers were what filled it, and
            # on a model where they are not, four retries at a few milliseconds
            # apart is a way of learning nothing: the run shrinks to a handful of
            # layers, still fails for the same reason, and reports that no memory
            # error explained it while every attempt was one. The numbers say
            # which side failed.
            _mem = _memory_report(device)
            if _mem:
                _why = _why + _mem
            # The model, the expert cache and the quantized dict are locals of
            # _load_and_run, so they go away with that frame and gc.collect()
            # has nothing to find. An earlier version tried to clear them out of
            # globals(), where they never were, which meant the collection was
            # the only thing standing between a failed attempt and the next one.
            # An even later version deleted the exception first, on the theory
            # that a caught exception pins its traceback; measured on this
            # interpreter, CPython does not, and the del changed nothing.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if len(_layers_now) <= 1:
                raise SystemExit(
                    f"Out of memory with a single trainable layer "
                    f"({_why}); the model does not fit this GPU at all."
                )
            _keep = max(1, int(len(_layers_now) * 0.75))
            print(f"\nOut of memory with {len(_layers_now)} trainable layers. "
                  f"Retrying with the top {_keep}.", flush=True)
            _layers_now = _layers_now[-_keep:]
    if model is None:
        # Every attempt above raised OutOfMemoryError, so saying no memory error
        # explained it denies the only thing that is known. The first attempt
        # failed for whatever reason, and every later one failed with less model
        # in it and no better outcome, which is the signature of a peak that the
        # trainable layers are not responsible for.
        raise SystemExit(
            "Out of memory on every attempt, with the layer count going "
            f"{_attempt_sizes} and ending at {len(_layers_now)}. "
            f"Last error: {_why}"
        )

    # --eval-report only ever did anything together with --eval-only, so a
    # training run accepted it, stored it, and wrote no report - the run that
    # actually produced the weights was the one that could not report on them.
    # Right after training is the only moment the report is worth writing.
    if config.eval_report:
        _write_eval_report(config, model, device)

    return model


def _write_eval_report(config, model, device):
    """Benchmark the model the way --eval-only does and save the report."""
    from usaf.eval.benchmark import BenchmarkConfig, run_benchmark
    from usaf.eval.report import save_report

    ds_list = [d.strip() for d in config.eval_datasets.split(",") if d.strip()]
    if not ds_list:
        raise SystemExit(
            "--eval-report was given but --eval-datasets is empty; there is "
            "nothing to report on"
        )

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    except Exception as e:
        raise SystemExit(
            f"--eval-report needs the tokenizer for {config.model_path}: {e}"
        )

    results = run_benchmark(
        model, tokenizer, device,
        BenchmarkConfig(
            datasets=ds_list,
            max_samples=config.eval_samples,
            seq_len=config.seq_len,
        ),
        model_name=config.model_path,
    )
    results.print()
    # save_report prints the path itself.
    save_report(results, config.eval_report)


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


def _apply_resume_overlays(resume_ckpt: dict, cache) -> dict[str, torch.Tensor]:
    """Scatter the trained weights from a checkpoint into the expert cache.

    Returns the active index map, which the training path needs as well.

    --eval-only used to skip this entirely: it ran the benchmark and returned
    before the resume block, so a model that had been trained down to a loss of
    1.37 benchmarked to exactly the same perplexity as the untouched base, and
    the report said so with a straight face. Evaluating a checkpoint you did not
    apply is worse than not evaluating at all.
    """
    active_idx = {k: v.to(torch.long) for k, v in resume_ckpt["active_idx"].items()}
    for fname, vals in resume_ckpt["masters"].items():
        p = torch.nn.Parameter(vals.float(), requires_grad=False)
        cache.overlays[fname] = (active_idx[fname].reshape(-1).to(torch.long), p)
    return active_idx

def _expert_modules_by_name(model, expert_modules):
    """Map expert module name to module, looking through a DataParallel wrapper.

    nn.DataParallel keeps the model as .module and prefixes every name it
    reports with "module.". Enumerating the wrapper therefore finds none of the
    expert modules, and the sparse-gradient hooks that are installed by name
    silently never get installed: the run trains nothing, the loss sits near
    ln(vocab), and the run still reports Complete. The wrapper shares the module
    objects, so unwrapping changes which names are looked up and nothing else -
    the attributes land on the same modules the forward pass uses.

    The count is checked because the failure mode is silence. A model whose
    experts are named differently from what detect_model reports has the same
    shape of problem, and a run that quietly learns nothing is worse than one
    that stops.
    """
    root = model.module if isinstance(model, nn.DataParallel) else model
    found = {n: m for n, m in root.named_modules() if n in expert_modules}
    if len(found) != len(expert_modules):
        missing = sorted(expert_modules - set(found))
        raise SystemExit(
            "no expert modules found: " + str(len(found)) + " of "
            + str(len(expert_modules)) + " matched under "
            + type(model).__name__ + ". First missing: " + str(missing[:3])
        )
    return found


def _expert_shapes(model, expert_modules: set[str]) -> dict[str, tuple]:
    """The expert tensors the model actually has, before anything clears them.

    Called before the cache hooks wipe ``_parameters``: after that the module
    carries whatever the q4 file said it should, which is exactly the thing
    that needs checking.
    """
    out: dict[str, tuple] = {}
    for mname, mod in model.named_modules():
        if mname in expert_modules:
            for pn, p in mod._parameters.items():
                out[f"{mname}.{pn}"] = tuple(p.shape)
    return out


def _validate_quant_weights(q_dict: dict, expected: dict[str, tuple], path: str) -> None:
    """Refuse a quantized file that does not describe this model.

    The expert parameters are cleared and refilled straight from the cache, and
    the training shapes are taken from the q4 file rather than from the model,
    so a file belonging to a different MoE loads without complaint: pointing
    tiny-moe at the Mixtral experts_q4.pt trained to a plausible loss while
    running on entirely the wrong weights, and reported success. Every tensor
    the model expects has to be present, at the shape the model has.
    """
    problems: list[str] = []
    for name, shape in expected.items():
        entry = q_dict.get(name)
        if entry is None:
            problems.append(f"{name}: not in the quantized file")
            continue
        got = tuple(entry["shape"]) if isinstance(entry, dict) else tuple(entry[3])
        if got != tuple(shape):
            problems.append(f"{name}: quantized {got}, model has {tuple(shape)}")
    if not problems:
        return
    shown = "\n".join(f"  {p}" for p in problems[:8])
    more = f"\n  ... and {len(problems) - 8} more" if len(problems) > 8 else ""
    raise SystemExit(
        f"quantized weights do not match the model: {path}\n{shown}{more}\n"
        "  Quantize this model before training it, or point --quant-path at "
        "the file that belongs to it."
    )

def _resolve_quant_path(quant_path: str, model_name: str) -> str:
    """Work out which file holds the quantized expert weights.

    --quant-path names a file, but a directory is the natural thing to pass and
    torch.load reports a directory as "PermissionError: [Errno 13] Permission
    denied", which reads like a lock rather than a wrong path. Accept the
    directory form, and say exactly what is expected when nothing is there.
    """
    if not quant_path:
        # Next to the model, not next to wherever the user happens to be.
        #
        # usaf.quantize writes <model>-q4/experts_q4.pt resolved from the model
        # directory it was pointed at. The trainer looked for a bare relative
        # name, so the two commands only agreed when the model was a directory
        # sitting in the current directory. Handed any other path - what the Kaggle
        # kernel does, and what anyone who types a path does - quantize succeeded
        # and train then said the file was not there.
        _mp = os.path.abspath(model_name)
        quant_path = os.path.join(os.path.dirname(_mp),
                                 os.path.basename(_mp) + "-q4",
                                 "experts_q4.pt")
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


def _rebuild_inv_freq(rotary_mod, cfg, layer_type: str | None = None) -> torch.Tensor:
    """Recompute a RoPE module inv_freq buffer from the model config.

    The model is materialised on the meta device, so every buffer - including
    inv_freq - starts out without real storage. We rebuild it with the same
    routine transformers itself uses (ROPE_INIT_FUNCTIONS), driven by the
    config, so the rotary dimension, theta and any rope_scaling are honoured
    instead of being guessed.

    ``layer_type`` exists for stacks that carry one RoPE per layer type. Those
    register the buffer under a name built from the type ("default_inv_freq",
    "hybrid_sliding_inv_freq") and look it up by that name in forward, so the
    buffer has to be rebuilt per type from that type's own parameters. Calling
    this with no layer_type covers the ordinary single-RoPE case and is what
    every other model needs.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    # rope_type lives in config.rope_parameters on transformers 5.x and in
    # config.rope_scaling (or top-level rope_theta) on 4.x. The "default"
    # type has no entry in ROPE_INIT_FUNCTIONS - it is computed by the rotary
    # class itself, so we dispatch to that static method when present.
    all_rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    if layer_type is None:
        rp = all_rp
        default_fn = getattr(type(rotary_mod), "compute_default_rope_parameters", None)
    else:
        sub = all_rp.get(layer_type)
        if sub is None:
            raise ValueError(
                f"layer type {layer_type!r} has no rope parameters; config has "
                f"{sorted(all_rp)}"
            )
        rp = sub
        fn = ROPE_INIT_FUNCTIONS.get(rp.get("rope_type", rp.get("type", "default")))
        if fn is not None:
            default_fn = fn
        else:
            cls_fn = getattr(type(rotary_mod), "compute_default_rope_parameters", None)
            default_fn = cls_fn if callable(cls_fn) else None
    rope_type = rp.get("rope_type", rp.get("type", "default"))
    if rope_type == "default" and callable(default_fn):
        # transformers 5.17 deprecates the device kwarg (removed in 5.18).
        # Inspect the signature so we neither emit a FutureWarning on 5.17
        # nor crash with a TypeError on 5.18+.
        import inspect as _inspect
        _params = _inspect.signature(default_fn).parameters
        kwargs = {}
        if layer_type is not None and "layer_type" in _params:
            kwargs["layer_type"] = layer_type
        if "device" in _params:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                inv_freq, attn_scaling = default_fn(cfg, **kwargs, device=torch.device("cpu"))
        else:
            inv_freq, attn_scaling = default_fn(cfg, **kwargs)
    else:
        if rope_type not in ROPE_INIT_FUNCTIONS:
            raise ValueError(f"unsupported rope_type {rope_type!r}")
        kwargs = {}
        _fn = ROPE_INIT_FUNCTIONS[rope_type]
        import inspect as _inspect
        if layer_type is not None and "layer_type" in _inspect.signature(_fn).parameters:
            kwargs["layer_type"] = layer_type
        inv_freq, attn_scaling = _fn(
            cfg, **kwargs, device=torch.device("cpu"), seq_len=None
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


class _DecoderAdapter:
    """Runs a decoder stack that does not match USAF's assumptions.

    USAF assumed every decoder layer takes a 4-D causal mask tensor, returns a
    bare hidden state, and shares one set of rotary embeddings with the whole
    stack. A stack that differs on any of those three is signalled by
    transformers with ``config.layer_types`` - a per-layer type list that exists
    precisely to say the layers are not alike. ZAYA uses it for full versus
    sliding attention, and there the rotary looks its frequencies up by layer
    type, attention wants a dict of named masks, and the layer returns
    (hidden, router_hidden) as a pair.

    Without this, a run gets all the way through downloading, detecting and
    quantizing the model and then dies in the first forward. Nothing up to that
    point says the architecture is unsupported, which is what makes it look like
    a crash rather than a gap.

    The adaptations are driven by config.layer_types rather than by a model name,
    so they apply to any stack transformers marks as heterogeneous.
    """

    def __init__(self, layers, rotary, cfg, hidden, pos_ids, mask):
        self.layers = layers
        self.hybrid = bool(getattr(cfg, "layer_types", None))
        self.layer_types = list(getattr(cfg, "layer_types", []) or [])
        if self.hybrid:
            self.pe = {}
            for lt in dict.fromkeys(self.layer_types):
                self.pe[lt] = rotary(hidden, position_ids=pos_ids, layer_type=lt)
            self.mask = {"causal": mask, "padding": None}
        else:
            self.pe = rotary(hidden, position_ids=pos_ids)
            self.mask = mask

    def layer_type(self, i: int) -> str | None:
        if not self.hybrid or i >= len(self.layer_types):
            return None
        return self.layer_types[i]

    def call(self, i: int, h, pos_ids):
        """Run layer i, picking the right mask and rotary and unwrapping output."""
        lt = self.layer_type(i)
        pe = self.pe.get(lt) if self.hybrid else self.pe
        out = self.layers[i](h, attention_mask=self.mask, position_ids=pos_ids,
                            position_embeddings=pe)
        # Routers that hand a running summary to the next layer return a pair.
        # The summary is only needed for auxiliary router losses, which the
        # sparse path does not use, so the hidden state is the tensor to carry.
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


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


def materialize_meta_tensors(model, cfg, device, model_path, expert_modules):
    '''Fill in every tensor a meta-device model was built with.

    A model built under torch.device('meta') has no storage anywhere, and USAF
    clears the expert parameters on purpose: they are refilled from the quantized
    cache by forward pre-hooks. Everything else has to be filled here, because
    the first forward to reach a meta tensor does arithmetic against a real one
    and dies with 'Tensor on device meta is not on the expected device', which
    names neither the tensor nor the module.

    Three kinds of tensor, three treatments, in this order:

    RoPE frequency buffers are rebuilt from the config, by the same code the
    model itself uses, including the per-layer-type naming a heterogeneous stack
    registers. The clone of the frequencies that sits next to them, under the
    same name with an _original_ suffix, is a copy and is filled from the primary
    it was cloned from.

    A tied output head is re-tied to the input embedding, because a tied config
    stores the weights once and a reader that goes by name finds nothing for the
    head.

    Buffers that are registered but not persisted are zero-filled and named: an
    aux-loss accumulator has no checkpoint entry and no reconstruction rule, and
    zeros are its true value before the first step. A PARAMETER in that position
    is not filled, because a zeroed weight trains and exports as a model that
    looks fine and is wrong; those stop the run with their names attached.

    Expert tensors are excluded throughout: finding them on meta is the design.
    '''
    for mn, mod in model.named_modules():
        for bn, b in list(mod._buffers.items()):
            if b is not None and b.device.type == "meta":
                # Stacks that carry one RoPE per layer type register the buffer
                # under a name built from that type - "default_inv_freq",
                # "hybrid_sliding_inv_freq" - and look it up by that name in
                # forward. Matching only the literal name "inv_freq" left every
                # one of them on meta, and the first forward died with
                # AttributeError on "None_inv_freq". Anything ending in
                # _inv_freq is a RoPE frequency buffer; the prefix names the
                # layer type it belongs to.
                # A stack with one RoPE per layer type registers two buffers per
                # type: "<type>_inv_freq" and a clone named
                # "<type>_original_inv_freq". transformers' own single-RoPE
                # modules use the same "original" suffix with no prefix. Either
                # way the "original" copy is a clone, not a per-type frequency,
                # so it is skipped and filled from the primary below.
                if bn == "inv_freq":
                    _lt = None
                elif bn == "original_inv_freq" or bn.endswith("_original_inv_freq"):
                    _lt = None
                    bn = None
                elif bn.endswith("_inv_freq"):
                    _lt = bn[: -len("_inv_freq")]
                else:
                    _lt = None
                    bn = None
                if bn is not None:
                    # Rebuild RoPE exactly the way transformers does, from
                    # the config. The previous code guessed the rotary dim
                    # with getattr(mod, "dim", getattr(mod, "head_dim", 128)),
                    # but rotary modules expose neither attribute, so it
                    # always fell through to a hardcoded 128. On a model
                    # whose real head_dim differs that produced an inv_freq
                    # of the wrong length, and the forward died with
                    # "The size of tensor a (8) must match tensor b (128)".
                    _inv = _rebuild_inv_freq(mod, cfg, _lt).to(device=device)
                    mod._buffers[bn] = _inv

    # transformers keeps a clone of the frequencies next to the primary one,
    # with or without a layer-type prefix, and dynamic/linear rope reads it. It
    # is a copy, so it is filled from the primary rather than rebuilt. This runs
    # over every module rather than inside the loop above, because the bare
    # name is exactly the case where no primary was rebuilt to nest it in.
    # Stripping the suffix off the bare name leaves nothing, so its primary is
    # simply inv_freq; a per-type name keeps its type in front.
    for _mn2, _mod2 in model.named_modules():
        for _bn2, _b2 in list(_mod2._buffers.items()):
            if _bn2 != "original_inv_freq" and not _bn2.endswith("_original_inv_freq"):
                continue
            if _b2 is None or _b2.device.type != "meta":
                continue
            _base = _bn2[: -len("_original_inv_freq")]
            _prim = (_base + "_inv_freq") if _base else "inv_freq"
            _src = _mod2._buffers.get(_prim)
            if _src is not None and _src.device.type != "meta":
                _mod2._buffers[_bn2] = _src.clone()


    # A config with tie_word_embeddings does not store lm_head.weight in the
    # checkpoint - it is the input embedding. ZAYA1-8B is tied, so lm_head came
    # out of the load on meta and the first forward died in the head. Re-tie it
    # from the embedding that was just loaded, which is what the tied config
    # means, rather than inventing a second copy of the weights.
    if getattr(cfg, "tie_word_embeddings", False):
        _emb = dict(model.named_parameters()).get("model.embed_tokens.weight")
        _head = None
        for _mn3, _mod3 in model.named_modules():
            if _mn3.endswith("lm_head"):
                _head = _mod3
                break
        if _emb is not None and _head is not None and \
                _head.weight.device.type == "meta":
            _head.weight = nn.Parameter(
                _emb.detach().clone().to(device=device), requires_grad=False)
            print("  tied lm_head.weight to the input embedding", flush=True)

    # Buffers that are registered but not persisted - an aux-loss accumulator
    # like a router's balancing bias is the common case - have no checkpoint
    # entry and no reconstruction rule. transformers builds them in __init__, and
    # __init__ is exactly what cannot run against a meta model. Zeros are the
    # faithful value for an accumulator that has never seen a step, so they are
    # filled and named. A parameter in the same position is NOT zero-filled: a
    # zeroed weight trains and exports as a model that looks fine and is wrong,
    # so that still stops the run.
    zeroed: list[str] = []
    for _mn4, _mod4 in model.named_modules():
        for _bn4, _b4 in list(_mod4._buffers.items()):
            if _b4 is None or _b4.device.type != "meta":
                continue
            _mod4._buffers[_bn4] = torch.zeros(
                _b4.shape, dtype=_b4.dtype, device=device)
            zeroed.append(_mn4 + "." + _bn4)
    if zeroed:
        uniq_z = sorted(set(zeroed))
        print(f"  {len(uniq_z)} non-persistent buffer(s) zero-filled "
              f"(not stored in the checkpoint): "
              + ", ".join(uniq_z[:3])
              + (f", ... and {len(uniq_z) - 3} more" if len(uniq_z) > 3 else ""),
              flush=True)

    # Anything still on meta at this point is a tensor the model was built with
    # and nothing ever filled in: not in the checkpoint, not an expert, and not
    # one of the RoPE buffers repaired above. The first forward then reaches it,
    # does arithmetic against a real tensor, and dies hundreds of lines later
    # with "Tensor on device meta is not on the expected device", which names
    # neither the tensor nor the module. ZAYA1-8B did exactly that on a Kaggle
    # T4, after a completely normal-looking load. Checking here costs one walk
    # and turns an obscure crash into a list of names.
    #
    # Expert tensors are excluded on purpose. They are cleared from the modules
    # below and refilled from the quantized cache by forward pre-hooks, so
    # finding them on meta is the design, not a defect.
    def _is_expert(_name):
        return any(_name == _em or _name.startswith(_em + ".") for _em in expert_modules)

    still_meta = [
        n for n, p in model.named_parameters()
        if p is not None and p.device.type == "meta" and not _is_expert(n)
    ]
    for _n, _m in model.named_modules():
        for _bn, _b in list(_m._buffers.items()):
            if _b is not None and _b.device.type == "meta" and not _is_expert(_n):
                still_meta.append(_n + "." + _bn)
        for _pn, _p in list(_m._parameters.items()):
            if _p is not None and _p.device.type == "meta" and not _is_expert(_n):
                still_meta.append(_n + "." + _pn)
    if still_meta:
        uniq = sorted(set(still_meta))
        raise SystemExit(
            f"{len(uniq)} tensors are still on the meta device after loading, so the"
            " first forward will fail:\n  " + "\n  ".join(uniq[:12])
            + (f"\n  ... and {len(uniq) - 12} more" if len(uniq) > 12 else "")
            + f"\n\nThey are not in {model_path} and are not expert weights,"
            " so nothing in the load path can fill them. This architecture has"
            " tensors USAF does not materialise."
        )


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
        except Exception as e:
            # The bare except used to substitute a Qwen3-MoE for whatever failed
            # to build, which turns "this transformers cannot load that model"
            # into "train a different model" - quietly, and with a loss curve
            # that looks like progress. The substitution is only meaningful when
            # the config really is a Qwen3-MoE, and then it is a name-resolution
            # convenience rather than a rescue. Anything else stops.
            mt = str(getattr(cfg, "model_type", "") or "").lower()
            if "qwen3_moe" not in mt:
                raise SystemExit(
                    f"transformers {__import__('transformers').__version__} "
                    f"cannot build "
                    f"{getattr(cfg, 'model_type', type(cfg).__name__)} from "
                    f"{config.model_path}, and it is not a Qwen3-MoE so there is "
                    f"nothing to fall back to: {type(e).__name__}: {e}"
                ) from e
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
    # Every non-expert weight used to be cast with .half(), whatever the model
    # was written in. A bf16 checkpoint - ZAYA1 declares dtype bfloat16 and
    # keeps its residual stream in fp32 by design - came back as fp16, and the
    # mismatch is what made it unusable. The run dtype is chosen once, up
    # front, and applies to the weights, the causal mask and the rotary path.
    run_dtype = resolve_dtype(getattr(config, "dtype", "auto"), device)
    # The 4-bit format is fp16 by construction: dequantize_4bit, the expert
    # cache and the mmapped readers all hand back float16, and the scales are
    # stored as fp16. Loading the dense weights in a different precision only
    # produces "expected m1 and m2 to have the same dtype". So a quantized run
    # is fp16, and asking for anything else is an error rather than a silent
    # half-applied setting. Unquantized runs are free to use bf16, which is
    # what a bf16-native model on a T4 should run as.
    if run_dtype != torch.float16:
        raise SystemExit(
            f"--dtype {getattr(config, 'dtype', 'auto')!r} is not supported yet: the "
            "expert path is float16 by construction. dequantize_4bit, the mmapped "
            "readers and the expert cache all return float16, and dense weights in "
            "another precision fail at the first expert matmul with a dtype "
            "mismatch. The 4-bit scales are stored as fp16 too.\n"
            "  Use --dtype fp16 (or auto). Making the expert path precision-parametric "
            "is what would unlock bf16, and it is a real change, not a flag."
        )
    for name in sorted(wf.keys()):
        if any(name.startswith(m + ".") for m in _expert_modules):
            continue
        if name not in mp:
            continue
        with safe_open(os.path.join(st_path, wf[name]), framework="pt") as sf:
            tensor = sf.get_tensor(name).to(run_dtype)
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

    print(f"  {n_loaded} non-expert params loaded")
    materialize_meta_tensors(model, cfg, device, config.model_path, _expert_modules)


    q_dict = torch.load(config.quant_path, map_location="cpu", weights_only=True)
    _validate_quant_weights(q_dict, _expert_shapes(model, _expert_modules), config.quant_path)
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

    return model, cache, q_dict, wf, st_path, router_params, cfg


def _run_training(config, moe_cfg, model, cache, q_dict, device, scaler,
                  train_samples, eval_samples, heldout_samples,
                  _train_names, _shapes, train_layers,
                  layers, embed, rotary, norm_fn, lm_head,
                  router_params=None,
                  model_cfg=None,
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

    # Filled in once _prelude exists, just below.
    FROZEN_CACHE = None

    # --log-dir was accepted and ignored. Resolve it to a concrete file once,
    # here, and only open it per step so nothing is created for a run that is
    # going to be interrupted before its first step finishes.
    LOG_PATH = None
    if config.log_dir:
        os.makedirs(config.log_dir, exist_ok=True)
        _tag = config.tag or "run"
        LOG_PATH = os.path.join(config.log_dir, f"train_{_tag}.jsonl")
        print(f"  Log: {LOG_PATH}")

    def _prelude(input_ids):
        hidden = embed(input_ids)
        s_len = hidden.shape[1]
        pos_ids = torch.arange(s_len, device=device).unsqueeze(0)
        # The rotary is built by the adapter, not here. A stack with one RoPE per
        # layer type needs a different pair per type, and computing a single
        # shared pair first would be work thrown away - and, for those stacks,
        # the wrong one.
        # The mask is built in the run dtype. It used to be hardcoded fp16,
        # which mismatches a bf16 model: the fill value is fine in bf16 (same
        # exponent range as fp32) but an fp16 mask added to bf16 activations
        # either promotes the whole block or fails, depending on the kernel.
        _dt = hidden.dtype
        mask = torch.triu(
            torch.full((s_len, s_len), torch.finfo(_dt).min, device=device, dtype=_dt),
            diagonal=1).unsqueeze(0).unsqueeze(0)
        adapter = _DecoderAdapter(layers, rotary, model_cfg, hidden, pos_ids, mask)
        return hidden, pos_ids, adapter, mask

    # Frozen activation cache for layers 0..DETACH_AT.
    #
    # --no-frozen-cache used to be a flag that changed nothing: the config field
    # was read, printed as "Frozen cache: True/False", and no cache was ever
    # built. Layers 0..DETACH_AT are frozen by construction (only layers above
    # DETACH_AT are trainable), so their output for a given sample does not
    # change during a run and can be computed once and reused.
    if config.use_frozen_cache:
        if DETACH_AT >= 0:
            from usaf.frozen_cache import build_frozen_cache

            _n = config.frozen_cache_n if config.frozen_cache_n > 0 else len(train_samples)
            _fc_train = train_samples[:_n]
            for _i, _s in enumerate(_fc_train):
                _s["_fidx"] = _i

            def _compute_hidden(s):
                _ids = torch.tensor(s["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
                _h, _pos, _ad, _mask = _prelude(_ids)
                for _i in range(DETACH_AT + 1):
                    _h = _ad.call(_i, _h, _pos)
                return _h

            _ck = os.path.join(config.checkpoint_dir or "checkpoints",
                               f"frozen_cache_d{DETACH_AT}.npy")
            FROZEN_CACHE = build_frozen_cache(
                # model.config does not survive nn.DataParallel: the wrapper is
                # a different object and forwards module calls, not attributes,
                # so a multi-GPU run died here asking the wrapper for a config.
                # The config is already in hand - _load_model read it from the
                # same place the weights came from - and it does not change
                # when the model gets wrapped.
                _fc_train, SEQ, model_cfg.hidden_size, DETACH_AT,
                config.model_path, _compute_hidden, _ck,
            )
            print(f"  Frozen cache: layers 0..{DETACH_AT} for {len(_fc_train)} samples")
        else:
            print("  Frozen cache skipped: no layer is frozen (train-from is 0)")

    def _head_loss(hidden, labels):
        h = norm_fn(hidden)
        logits = lm_head(h)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    evals = []

    @torch.no_grad()
    def _eval_ppl(samples, max_n=6):
        """Token-weighted perplexity over the first ``max_n`` held-out samples.

        Runs a full forward rather than the training path: evaluation samples
        are not in the frozen cache, and correctness of the reported number
        matters more than the speed of printing it.
        """
        if not samples:
            return None
        total_loss = 0.0
        total_tok = 0
        for s in samples[:max_n]:
            ids = torch.tensor(s["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
            lbl = torch.tensor(s["labels"], dtype=torch.long).unsqueeze(0).to(device)
            h, pos, ad, mask = _prelude(ids)
            for i in range(N_LAYERS):
                h = ad.call(i, h, pos)
            n_tok = int((lbl[:, 1:] != -100).sum().item())
            if n_tok:
                total_loss += _head_loss(h, lbl).item() * n_tok
                total_tok += n_tok
            cache.evict_all()
        if total_tok <= 0:
            return None
        return math.exp(total_loss / total_tok)

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
        hidden, pos_ids, ad, mask = _prelude(ids)
        # Under no_grad, like the training forward. The loop below replays every
        # layer from xs_imp in reverse and threads the gradient through by hand,
        # so the graph this loop used to build was never used for anything - it
        # only pinned one layer worth of expert weights per layer, all at once,
        # because autograd keeps a matmul weight alive until that matmul is
        # differentiated. On a model small enough to fit, that is invisible. On
        # one with eleven trainable layers it is the whole card, and it is why
        # the importance pass ran out of memory on every layer count including
        # two: the peak had nothing to do with how many layers were trainable,
        # because it was always every layer at once.
        xs_imp = []
        with torch.no_grad():
            for i in range(N_LAYERS):
                xs_imp.append(hidden)
                hidden = ad.call(i, hidden, pos_ids)
        cache.evict_all()
        h_last = hidden.detach().requires_grad_(True)
        loss = _head_loss(h_last, lbl)
        loss.backward()
        g_imp = h_last.grad
        for j in range(len(xs_imp) - 1, -1, -1):
            x2 = xs_imp[j].detach().requires_grad_(True)
            out = ad.call(j, x2, pos_ids)
            out.backward(g_imp)
            g_imp = x2.grad
            cache.evict_all()
        return loss.item()

    def _do_export():
        #"""Write the merged model."""
        #
        # This has to be callable from the resume path as well. A run resumed
        # at or past its final step returns early, because there is nothing to
        # train, and that return used to sit above this block - so "train on one
        # machine, come back, export the merged model" produced no file and no
        # error. Training a checkpoint and exporting it is the whole point of
        # the second command.
        if not config.export_path:
            return
        print(f"\nExporting merged weights to {config.export_path}...")
        # A failed export used to print a line and let the run finish, so the
        # exit code was 0 and the report said Complete. The export is the entire
        # reason for the second command in "train on one machine, export on
        # another" - a run that trains for hours and then quietly produces no
        # model is worse than one that stops, because nothing tells you to go
        # looking. config.export_path being set means the user asked for it, so a
        # failure here is fatal.
        from usaf.checkpoint import export_merged_weights

        try:
            export_path = export_merged_weights(
                config.quant_path,
                {k: v.detach().cpu() for k, v in masters.items()},
                {k: v.cpu() for k, v in active_idx.items()},
                config.export_path,
            )
        except Exception as e:
            raise SystemExit(
                f"--export was requested but the merge failed: "
                f"{type(e).__name__}: {e}"
            ) from e
        print(f"  Exported: {export_path}")
    if resume_ckpt is None:
        imp_store = TopKImportanceStore(_shapes, frac=FRAC)
        for mname, mod in _expert_modules_by_name(
                model, _expert_modules).items():
            mod._grad_capture = (imp_store, mname)

        print("Importance phase...")
        t0 = time.time()
        N_IMP = 3 if not os.environ.get("SMOKE_N") else 1
        for imp_i in range(N_IMP):
            s = train_samples[imp_i % len(train_samples)]
            loss_imp = fwd_bwd_imp(s)
            print(f"  imp {imp_i+1}/{N_IMP} | loss {loss_imp:.4f} | {time.time()-t0:.0f}s")

        active_idx = imp_store.select(FRAC)
        # Release the candidates now that they have been ranked. The store keeps
        # cand_mult times the budget for every expert slice of every tensor, and
        # for ZAYA that is 805M values in float32 plus the same number of int64
        # indices: 9.66 GB, held for the rest of the run by a local that nothing
        # else refers to. The first ZAYA run with a real active set was killed
        # by the OOM killer at the step-50 reselect with 20.1 GB already used and
        # this 9.66 GB sitting unused beside it.
        imp_store.zero_()
        # A run that selects nothing is a run that trains no experts, and every
        # other signal in it says otherwise: the architecture prints, the
        # quantiser writes the tensors, the loss falls, the kernel reports
        # Complete. ZAYA was trained for an hour that way. The reason is always
        # the same - the capture lives inside the patched expert forward, and a
        # model whose expert container nothing patched runs the stock forward,
        # which has no hook. Patching one family at a time cannot keep up with
        # that; refusing to continue can.
        if not active_idx:
            _n_mods = len(_expert_modules_by_name(model, _expert_modules))
            raise SystemExit(
                f"the importance pass captured nothing, so no expert weight "
                f"would be trained. {_n_mods} expert modules were found at "
                f"{sorted(_expert_modules)[:2]}, and the sparse-gradient hooks "
                f"that would have filled them run inside the replaced expert "
                f"forward - so this model's expert container is not one USAF "
                f"patches. The loss would still fall, on the non-expert "
                f"parameters alone."
            )
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
        active_idx = _apply_resume_overlays(resume_ckpt, cache)
        masters = {
            fname: cache.overlays[fname][1] for fname in resume_ckpt["masters"]
        }
        losses = list(resume_ckpt.get("losses", []))
        start_step = resume_ckpt.get("step", 0) + 1
        if start_step > STEPS:
            print(f"Checkpoint step {start_step-1} >= total steps {STEPS}, nothing to train")
            # still export: coming back to a finished run to produce the
            # merged model is the ordinary reason to resume it at all.
            _do_export()
            return losses

    sparse_store = SparseGradStore(active_idx, _shapes)
    for mname, mod in _expert_modules_by_name(
            model, _expert_modules).items():
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
        hidden, pos_ids, ad, mask = _prelude(ids)

        with torch.no_grad():
            _cached = FROZEN_CACHE is not None and all("_fidx" in s for s in batch)
            if _cached:
                from usaf.frozen_cache import get_hidden
                hidden = torch.cat(
                    [get_hidden(FROZEN_CACHE, s["_fidx"], device) for s in batch],
                    dim=0).to(hidden.dtype)
            else:
                for i in range(DETACH_AT + 1):
                    hidden = ad.call(i, hidden, pos_ids)
            cache.evict_all()
            xs = []
            for i in range(DETACH_AT + 1, N_LAYERS):
                xs.append(hidden)
                hidden = ad.call(i, hidden, pos_ids)
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
            out = ad.call(i, x2, pos_ids)
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
    def do_reselect(_step):
        """RigL: one dense importance pass, then merge top-k into the active set.

        The universal CLI kept --reselect-every, printed "RigL: every N steps"
        and wrote the number into the checkpoint, but never reselected: the
        variable was read once and used once more, to record itself. Only the
        standalone root train.py had this. A name on a flag is not behaviour.
        """
        nonlocal active_idx, masters, sparse_store, opt
        t0 = time.time()
        print(f"  [reselect step {_step}] dense importance pass...", flush=True)

        _imp = TopKImportanceStore(_shapes, frac=FRAC)
        for _m, _mod in _expert_modules_by_name(
                model, _expert_modules).items():
            _mod._grad_capture = (_imp, _m)

        fwd_bwd_imp(train_samples[0])
        new_idx = _imp.select(FRAC)
        _imp.zero_()
        del _imp

        _new_active = {}
        _new_masters = {}
        _kept_n = _dropped_n = _grown_n = 0

        for fname, old in active_idx.items():
            nw = new_idx.get(fname)
            if nw is None or nw.numel() == 0:
                _new_active[fname] = old.clone()
                _new_masters[fname] = masters[fname]
                continue
            # Set arithmetic in Python over a 110M-parameter active set. Each
            # tensor holds millions of indices, and a set of Python ints costs
            # tens of bytes apiece, so the intersection and both sorted lists
            # were hundreds of megabytes per tensor and minutes of wall clock.
            # The same three operations are one unique and two searchsorted.
            _old_uniq = old.reshape(-1).to(torch.long).unique()
            _nw_uniq = nw.reshape(-1).to(torch.long).unique()
            _is_kept = torch.isin(_nw_uniq, _old_uniq)
            kept_idx = _nw_uniq[_is_kept]
            grown_idx = _nw_uniq[~_is_kept]
            fill = max(0, _old_uniq.numel() - kept_idx.numel())
            final = torch.cat([kept_idx, grown_idx[:fill]])
            _new_active[fname] = final

            _old_vals = masters[fname].data.float()
            _old_flat = old.reshape(-1).to(torch.long)
            _keep_mask = torch.isin(final, _old_flat)
            _kept_n += int(_keep_mask.sum().item())
            _dropped_n += _old_flat.numel() - int(
                torch.isin(_old_flat, final).sum().item())
            _grown_n += int((~_keep_mask).sum().item())

            _new_vals = torch.zeros(final.numel(), dtype=torch.float32)
            _kept_pos = _keep_mask.nonzero(as_tuple=False).reshape(-1)
            if _kept_pos.numel() > 0:
                _all_idx = torch.zeros(
                    int(_old_flat.max().item()) + 1, dtype=torch.long)
                _all_idx[_old_flat] = torch.arange(
                    _old_flat.numel(), dtype=torch.long)
                _new_vals[_kept_pos] = _old_vals[_all_idx[final[_kept_pos]]]

            _grow_pos = (~_keep_mask).nonzero(as_tuple=False).reshape(-1)
            if _grow_pos.numel() > 0:
                entry = q_dict[fname]
                if isinstance(entry, dict) and 'q' in entry:
                    t = dequantize_4bit(
                        entry['q'], entry['s'], entry['z'], entry['shape'],
                        group_size=128)
                else:
                    t = dequantize_4bit(
                        entry[0], entry[1], entry[2], entry[3], group_size=128)
                _new_vals[_grow_pos] = t.reshape(-1)[final[_grow_pos]].float()
                del t
            _new_masters[fname] = torch.nn.Parameter(
                _new_vals, requires_grad=False)

        active_idx = _new_active
        masters = _new_masters
        sparse_store = SparseGradStore(active_idx, _shapes)
        for _m, _mod in _expert_modules_by_name(
                model, _expert_modules).items():
            _mod._grad_capture = (sparse_store, _m)

        cache.overlays.clear()
        for fname, aidx in active_idx.items():
            cache.overlays[fname] = (
                aidx.reshape(-1).to(torch.long), masters[fname])
        if USE_RESIDENT:
            cache.apply_resident_overlays(active_idx, masters)
        # Carry the Adam moments across the reselection. Building a fresh
        # SparseAdam here threw m and v away every reselect_every steps
        # and rewound the step counter, and the checkpoint then recorded
        # that rewound number: resuming a run that had reselected came
        # back saying step 1 when the run was at step 3. reselect() maps
        # each surviving element back to its old moment and leaves newly
        # activated ones clean.
        _lr_keep = opt.lr
        opt.reselect(masters, active_idx)
        opt.lr = _lr_keep
        _ta = sum(i.numel() for i in active_idx.values())
        _te = sum(math.prod(_shapes[fn]) for fn in active_idx if fn in _shapes)
        print(f"  [reselect] kept={_kept_n:,} dropped={_dropped_n:,} "
              f"grown={_grown_n:,} active={_ta:,} "
              f"({100*_ta/max(_te,1):.4f}%) in {time.time()-t0:.0f}s",
              flush=True)
    for step in range(start_step, STEPS + 1):
        t_step = time.time()

        if step <= max(1, int(STEPS * 0.05)):
            lr = LR_PEAK * step / max(1, int(STEPS * 0.05))
        else:
            progress = (step - max(1, int(STEPS * 0.05))) / max(1, STEPS - max(1, int(STEPS * 0.05)))
            lr = LR_PEAK * 0.1 + LR_PEAK * 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        opt.lr = lr

        if RESELECT_EVERY > 0 and step > 1 and step % RESELECT_EVERY == 0:
            do_reselect(step)
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

        # --eval-every was accepted and then ignored, so a run reported nothing
        # about its own progress until the very end. On a long job that is the
        # difference between seeing the loss go sideways at step 40 and finding
        # out at step 400. The evaluation runs after the step has been applied,
        # so it reflects the current weights.
        if config.eval_every > 0 and step % config.eval_every == 0 and eval_samples:
            print(f"  [eval {step}] running...", flush=True)
            _pp = _eval_ppl(eval_samples)
            if _pp is not None:
                evals.append((step, _pp))
                print(f"  [eval {step}] ppl={_pp:.2f}", flush=True)
                if LOG_PATH:
                    with open(LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, "eval_ppl": round(_pp, 4)}) + "\n")

        # --log-dir used to be accepted and then ignored, so a run left no
        # machine-readable trace of what it did. One JSON object per step,
        # appended as the run goes, so a long job can be inspected while it is
        # still going instead of only from the final summary.
        if LOG_PATH:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": step, "loss": step_loss, "lr": lr,
                    "scale": loss_scale, "finite": finite,
                    "sec": round(dt, 2), "ram": round(ram(), 2),
                    "tok_s": round(tok_s, 1),
                }) + "\n")

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


    # A loss curve is evidence that something moved, not that anyone can tell.
    # The question a person actually has after a fine-tune is what the model now
    # says, and answering it meant reloading 17 GB of base weights in a second
    # process. In here the model is already built and the trained values are
    # already in the cache overlays, so the only thing missing was the call.
    #
    # The expert modules had their parameters cleared at load time and are
    # refilled by forward pre-hooks from the quantized cache, which is exactly
    # what makes a plain model(...) work here: the graph goes through the same
    # path the training forward took, with the overlays already applied.
    if config.generate_after:
        try:
            tokenizer = _get_tokenizer(config.model_path)
            if tokenizer is None:
                print("\n[generate] no tokenizer for this model, skipping",
                      flush=True)
            else:
                # DataParallel wraps the module but does not forward generate, so the unwrapped model is what samples.
                _gen_model = model.module if hasattr(model, "module") else model
                _gen_model.eval()
                for _prompt in config.generate_after.split("||"):
                    _prompt = _prompt.strip()
                    if not _prompt:
                        continue
                    _inputs = tokenizer(_prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        _out = _gen_model.generate(
                            **_inputs,
                            max_new_tokens=config.generate_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            # pad_token_id 0 is a real id, not a missing one.
                            # `or` treated it as absent and filled with eos
                            # instead, which for a tokenizer that uses 0 is
                            # the one value the model has certainly seen.
                            pad_token_id=(
                                tokenizer.pad_token_id
                                if tokenizer.pad_token_id is not None
                                else tokenizer.eos_token_id
                            ),
                        )
                    _new = _out[0][_inputs["input_ids"].shape[1]:]
                    print()
                    print("=" * 66)
                    print(f"PROMPT: {_prompt}")
                    print("-" * 66)
                    print(tokenizer.decode(_new, skip_special_tokens=True))
                    print("=" * 66, flush=True)
                _gen_model.train()
        except Exception as _gen_err:  # noqa: BLE001
            # Generation is a demonstration, not the point of the run. Losing it
            # must not turn a finished training run into a failed one, but it
            # must not be silent either: the reason goes to stdout.
            print(f"\n[generate] failed: {type(_gen_err).__name__}: {_gen_err}",
                  flush=True)
    t_total = time.time() - t_start
    skipped = sum(1 for l in losses if not math.isfinite(l))

    print("\n=== Complete ===")
    print(f"Time: {t_total/3600:.1f}h")
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"Skipped: {skipped}/{STEPS} steps")
    print(f"Peak RAM: {ram():.1f}GB")

    _do_export()

    return losses


if __name__ == "__main__":
    main()
