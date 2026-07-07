"""Expert streaming for OLMoE (13.8GB) on a 12GB GPU.

Non-expert weights stay GPU-resident. Expert weights (~805MB/layer) stream from CPU
via forward hooks; gradient checkpointing (reentrant) keeps ≤1 expert layer in VRAM.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _is_expert_param(name: str) -> bool:
    return ".mlp.experts." in name


def _move_module_tensors(module: nn.Module, target: torch.device) -> None:
    """Move a single module's params and buffers to *target*, reassigning Parameter
    objects instead of touching ``.data``.
    
    DML rejects ``param.data = x.to(dev)`` (incompatible tensor type) in both
    directions. Reassigning the Parameter in ``module._parameters`` is the only
    reliable path.  ``.grad`` is carried over on the same device as ``.data``.
    """
    for pname, p in list(module._parameters.items()):
        if p is None or p.device == target:
            continue
        new = torch.nn.Parameter(p.detach().to(target), requires_grad=p.requires_grad)
        if p.grad is not None:
            new.grad = p.grad.to(target)
        module._parameters[pname] = new

    for bname, b in list(module._buffers.items()):
        if b is None or b.device == target:
            continue
        module._buffers[bname] = b.to(target)


def setup_streaming(
    model: nn.Module,
    device: torch.device,
    verbose: bool = True,
) -> nn.Module:
    """Move non-expert params to GPU, keep experts on CPU, install streaming hooks."""
    cpu = torch.device("cpu")

    n_gpu = n_cpu = 0
    bytes_gpu = bytes_cpu = 0
    for mname, mod in model.named_modules():
        is_expert = mname.endswith(".mlp.experts")
        target = cpu if is_expert else device
        for pname, p in list(mod._parameters.items()):
            if p is None:
                continue
            nbytes = p.numel() * p.element_size()
            if is_expert:
                n_cpu += 1
                bytes_cpu += nbytes
            else:
                n_gpu += 1
                bytes_gpu += nbytes
        _move_module_tensors(mod, target)

    if verbose:
        print(f"  GPU (residente): {n_gpu} tensores, {bytes_gpu / 1e9:.2f}GB")
        print(f"  CPU (streamed):  {n_cpu} tensores, {bytes_cpu / 1e9:.2f}GB")

    expert_modules: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if name.endswith(".mlp.experts"):
            expert_modules.append((name, mod))

    state: dict[str, int] = {"resident": 0, "max": 0}

    if not hasattr(model, '_expert_grads'):
        model._expert_grads: dict[str, torch.Tensor] = {}
    grad_store = model._expert_grads

    def make_pre_hook(mod_name: str):
        def pre_hook(module: nn.Module, _args: list) -> None:
            canon: dict[str, nn.Parameter] = {}
            for pname, p in list(module._parameters.items()):
                if p is None:
                    continue
                canon[pname] = p
                dev_p = torch.nn.Parameter(
                    p.detach().to(device), requires_grad=p.requires_grad
                )
                module._parameters[pname] = dev_p
                if dev_p.requires_grad:
                    full = f"{mod_name}.{pname}"

                    def _cap(param: nn.Parameter, _full: str = full) -> None:
                        if param.grad is not None:
                            grad_store[_full] = param.grad.detach()

                    dev_p.register_post_accumulate_grad_hook(_cap)
            module._canon = canon
            state["resident"] += 1
            state["max"] = max(state["max"], state["resident"])
            return None

        return pre_hook

    def make_post_hook():
        def post_hook(module: nn.Module, _args: list, output: torch.Tensor):
            canon = getattr(module, "_canon", None)
            if canon:
                for pname, p in canon.items():
                    module._parameters[pname] = p
                module._canon = None
            state["resident"] -= 1
            return output

        return post_hook

    model._stream_state = state

    for name, mod in expert_modules:
        mod.register_forward_pre_hook(make_pre_hook(name))
        mod.register_forward_hook(make_post_hook())

    if verbose:
        print(f"  hooks de streaming em {len(expert_modules)} modulos de experts")

    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
        if verbose:
            print("  gradient checkpointing: enabled (reentrant)")
    except Exception as e:
        if verbose:
            print(f"  gradient checkpointing: falhou ({e})")

    return model


def apply_captured_expert_grads(
    model: nn.Module, scale: float = 1.0
) -> int:
    """Copy captured expert grads to the canonical Parameters so the optimizer
    sees them via ``param.grad``.

    Must be called after backward and before ``optimizer.step`` (and before
    any ``optimizer.refresh`` that re-binds ``_named`` to current Parameter
    objects).
    """
    store: dict[str, torch.Tensor] = getattr(model, "_expert_grads", {})
    if not store:
        return 0

    params = dict(model.named_parameters())
    applied = 0

    for full, g in store.items():
        p = params.get(full)
        if p is not None:
            gc = g.detach().to("cpu").float() * scale
            if p.device.type == "cpu":
                p.grad = gc.to(dtype=p.dtype)
            else:
                p.grad = gc.to(device=p.device, dtype=p.dtype)
            applied += 1

    store.clear()
    return applied


def sync_expert_grads_to_cpu(model: nn.Module) -> int:
    """Backward-compatible alias — B1 called this; now delegates to
    ``apply_captured_expert_grads``."""
    return apply_captured_expert_grads(model)
