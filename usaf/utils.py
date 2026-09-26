from typing import Any

import torch

# torch_directml is imported here no more. A torch/directml mismatch fails in
# the Windows loader, not in Python: ImportError is raised from a DLL that
# does not have the entry point, and when faulthandler is active - which it is
# under pytest, and under python -X faulthandler - the process dies there with
# a native STATUS_ENTRYPOINT_NOT_FOUND and no Python-level error at all. It
# did that to the test suite on a machine where nothing was going to use
# DirectML: importing usaf was enough. The import now happens only where a DML
# device is actually asked for.
_dml_module = None
_dml_reason = ""
_dml_checked = False


def _declared_torch_requirement() -> str:
    """The torch pin torch-directml declares, or "" when it declares none."""
    try:
        from importlib.metadata import PackageNotFoundError, requires

        for req in requires("torch-directml") or []:
            name, _, spec = req.partition("==")
            if name.strip().lower() == "torch" and spec:
                return spec.strip()
    except (PackageNotFoundError, Exception):  # noqa: BLE001 - best effort
        return ""
    return ""


def _dml():
    """Import torch_directml on first use. Returns None when it cannot load.

    The version pin is checked before the import, and that is the part that
    matters. torch-directml is a compiled extension pinned to one exact torch
    build; a mismatch fails inside the Windows loader, and the resulting
    ImportError is raised while faulthandler is unwinding a native frame - so
    it can take the whole interpreter down with STATUS_ENTRYPOINT_NOT_FOUND
    and no Python traceback, and `except ImportError` never gets to run. Here
    torch-directml 0.2.5.dev240914 wants torch==2.4.1 against an installed
    2.13.0, which is knowable from the metadata alone, with no native load.
    """
    global _dml_module, _dml_reason, _dml_checked
    if _dml_checked:
        return _dml_module
    _dml_checked = True

    want = _declared_torch_requirement()
    if want:
        have = torch.__version__.split("+")[0]
        # Compare the numeric prefix only: 2.4.1 vs 2.4.1+cu121 is a match.
        have_num = ".".join(have.split(".")[: len(want.split("."))])
        if have_num != want:
            _dml_module = None
            _dml_reason = (
                f"torch-directml needs torch=={want}, installed {have}"
            )
            return None

    try:
        import torch_directml as _mod

        _dml_module = _mod
    except Exception as e:  # noqa: BLE001 - a broken install must not be fatal
        _dml_module = None
        _dml_reason = str(e) or type(e).__name__
    return _dml_module


def dml_unavailable_reason() -> str:
    """Why DirectML cannot be used, or "" when it can."""
    _dml()
    return _dml_reason


def has_dml() -> bool:
    """Whether DirectML can be loaded, without raising."""
    return _dml() is not None


def get_dml_device(device_id: int = 0) -> torch.device:
    mod = _dml()
    if mod is None:
        why = dml_unavailable_reason() or "not installed"
        raise RuntimeError(
            f"DirectML is unavailable: {why}. Use CUDA or CPU instead."
        )
    return mod.device(device_id)


def load_model_to_dml(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model.to(device)
    return model


def get_cpu_device() -> torch.device:
    return torch.device("cpu")


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """The dtype a run should use when the caller did not choose one.

    This used to return float32 for every backend except DirectML, including
    CUDA. On a T4 or P100 that is the worst available answer: it doubles the
    weight footprint against fp16/bf16 and roughly halves throughput, and for
    an 8B model it is the difference between fitting on a 16GB card and not.
    Nothing called it, so the choice was never actually made in the first
    place.

    bf16 is preferred over fp16 on hardware that has it. Its range matches
    fp32, so models that carry fp32 residuals - ZAYA1 keeps its residual
    stream in fp32 on purpose - do not overflow when a weight is loaded in a
    narrower dtype.
    """
    if device.type == "privateuseone":
        # DirectML has no bf16 kernels; fp16 is the only reduced precision there.
        return torch.float16
    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Turn a ``--dtype`` value into the dtype a run will actually use.

    ``auto`` resolves to float16, not to whatever the hardware would prefer,
    because the expert path constrains it: ``dequantize_4bit``, the mmapped
    readers and the expert cache all produce float16, and a run whose dense
    weights are a different precision dies at the first expert matmul with
    "expected m1 and m2 to have the same dtype". Claiming bf16 here because the
    GPU supports it would be a flag that lies.

    The explicit values are still accepted, and a run that asks for one the
    expert path cannot provide is rejected up front with that explanation
    rather than crashing later.
    """
    key = (name or "auto").strip().lower()
    named = {
        "auto": torch.float16,
        "": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if key not in named:
        raise SystemExit(
            f"--dtype {name!r} is not a precision. "
            "Use one of: auto, fp16, bf16, fp32."
        )
    return named[key]


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_optimizer_memory(num_active_params: int, dtype: torch.dtype = torch.float32) -> int:
    bytes_per_param = 4 if dtype == torch.float32 else 2
    return num_active_params * bytes_per_param * 2


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def dense_router_weights(
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    skip_index: int | None = None,
) -> torch.Tensor:
    """Expand a top-k routing decision into a dense [tokens, experts] mask.

    transformers >= 4.53 hands the expert container two tensors -
    ``top_k_index`` [N, k] and ``top_k_weights`` [N, k] - instead of the single
    pre-combined weight matrix that the DirectML expert loops were written
    against. They were therefore called with three positional arguments and
    declared two, so every DirectML run died with a TypeError on the first
    expert forward.

    ``skip_index`` is the "no expert" slot some routers reserve as one past
    the last expert. A token routed there contributes nothing rather than
    indexing a nonexistent expert.
    """
    n_tokens = top_k_index.shape[0]
    out = torch.zeros(
        n_tokens,
        num_experts,
        dtype=top_k_weights.dtype,
        device=top_k_weights.device,
    )
    if skip_index is not None:
        top_k_weights = top_k_weights.masked_fill(
            top_k_index == skip_index, 0.0
        )
    out.scatter_(1, top_k_index.clamp(max=num_experts - 1), top_k_weights)
    return out
