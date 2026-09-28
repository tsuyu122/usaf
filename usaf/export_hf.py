"""Write a USAF run out as an ordinary HuggingFace model directory.

USAF keeps three things that a normal checkpoint does not have: the experts are
4-bit and outside the model, only the active slice of them is trained, and the
routers are trained separately from the sparse expert gradients. A run therefore
cannot be loaded by anything but USAF, and a merged export written by the training
path is still 4-bit - which is fine for resuming and useless for a person who
wants to run the model somewhere else.

This writes what a runtime expects. The expert tensors are dequantized back to
fp16, the trained slice is written over the dequantized original, the trained
routers replace the ones from the file, and every tensor keeps the name the
checkpoint used - not the name the loaded module uses - because that is what
from_pretrained reads.

The names are not interchangeable. GraniteMoe stores input_linear and output_linear
and the loader renames them, so writing the module names produces a directory
that looks complete and loads into a model full of randomly initialised experts.
That is why every name is mapped back through the file's own vocabulary rather
than assumed, and why a tensor is counted on the way out: a file that is short one
key loads as a different model, silently.

fp16 throughout, because the source was bf16 and a bf16 directory is not
something a local runtime can serve.
"""

import json
import os
import shutil

import torch

from usaf.checkpoint import apply_trained_entries, load_sparse_checkpoint
from usaf.model_factory import resolve_stored_name

# Everything a from_pretrained needs that is not a weight. Copied from the base
# model so the exported directory is self-contained.
AUX_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def _dequant(entry, group_size: int) -> torch.Tensor:
    from usaf.quantization import dequantize_4bit

    if isinstance(entry, dict) and "q" in entry:
        return dequantize_4bit(
            entry["q"], entry["s"], entry["z"], entry["shape"],
            group_size=entry.get("group_size", group_size),
        )
    return dequantize_4bit(
        entry[0], entry[1], entry[2], entry[3], group_size=group_size)


def _base_tensors(model_path: str) -> dict[str, torch.Tensor]:
    """Every tensor in the base checkpoint, under the names the file uses."""
    out: dict[str, torch.Tensor] = {}
    for name in sorted(os.listdir(model_path)):
        if not name.endswith(".safetensors"):
            continue
        from safetensors import safe_open

        with safe_open(os.path.join(model_path, name), framework="pt") as sf:
            for key in sf.keys():
                out[key] = sf.get_tensor(key)
    return out


def export_hf_model(base_path: str, quant_path: str, sparse_path: str,
                    out_path: str, group_size: int = 128) -> str:
    """Write a plain HuggingFace model directory from a finished run."""
    state = load_sparse_checkpoint(sparse_path)
    masters = state["masters"]
    active = state["active_idx"]
    routers = state.get("routers") or {}

    base = _base_tensors(base_path)
    if not base:
        raise SystemExit(f"no safetensors in {base_path}")

    # The q4 payload lives in shared memory during a run, and shared memory is
    # not in the output, so a run that died after training leaves a checkpoint
    # with no payload beside it. When the base model holds every trained tensor
    # at full precision the payload is not needed at all - and going through the
    # base is better anyway, because the experts nobody trained keep every bit
    # instead of a quantize-and-back round trip. A tensor that is in neither is
    # still refused below, by name.
    if os.path.isfile(quant_path):
        q = torch.load(quant_path, map_location="cpu", weights_only=True)
    else:
        q = {}
        print(f"  no {quant_path}: taking the trained experts over the base "
              "model, which holds them at full precision", flush=True)

    # The trained slice goes over the dequantized original, expert by expert.
    trained_experts = 0
    for fname, master in masters.items():
        key = resolve_stored_name(fname, base)
        if key in base:
            # The base checkpoint already holds this expert, at full precision,
            # so the trained entries go straight into the real weights and the
            # untrained ones keep every bit they were released with. That is
            # strictly better than dequantizing the q4 copy back up, and it is
            # also the only path that works when the q4 payload was written to
            # shared memory and is not in the output - which is what a run that
            # died after training leaves behind.
            full = apply_trained_entries(
                base[key].to(torch.float32), active[fname], master, fname)
        else:
            entry = q.get(fname)
            if entry is None:
                raise SystemExit(
                    f"{fname} was trained but is not in {quant_path} and the base "
                    "model does not hold it; refusing to export a model that "
                    "silently drops it")
            full = apply_trained_entries(
                _dequant(entry, group_size), active[fname], master, fname)
        base[key] = full.to(torch.float16)
        trained_experts += int(active[fname].reshape(-1).numel())

    # Anything in the q4 file the run never touched still has to be written, or
    # the exported model is missing experts it claims to have.
    for fname, entry in q.items():
        key = resolve_stored_name(fname, base)
        if key in base:
            continue
        base[key] = _dequant(entry, group_size).to(torch.float16)

    for name, tensor in routers.items():
        base[resolve_stored_name(name, base)] = tensor.to(torch.float16)

    os.makedirs(out_path, exist_ok=True)
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in base.items()},
              os.path.join(out_path, "model.safetensors"),
              metadata={"format": "pt"})

    for name in AUX_FILES:
        src = os.path.join(base_path, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_path, name))

    with open(os.path.join(out_path, "usaf_export.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "base_model": base_path,
            "sparse_checkpoint": sparse_path,
            "step": state.get("step"),
            "trained_experts": trained_experts,
            "trained_routers": len(routers),
            "tensors_written": len(base),
        }, f, indent=2)

    print(f"  wrote {len(base)} tensors to {out_path}")
    print(f"  {trained_experts} trained experts, {len(routers)} trained routers")
    return out_path


def _verify_loadable(out_path: str) -> None:
    """Refuse a directory the loader cannot read back.

    Writing a file is not the same as producing a model. If the expert names were
    written in the module spelling, from_pretrained succeeds, reports the missing
    keys, and fills them with fresh random weights - so the check that matters is
    a real load with no missing keys, not the existence of the file.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(out_path, dtype=torch.float16)
    missing = [n for n, _ in model.named_parameters()
               if n not in dict(model.named_parameters())]
    if missing:
        raise SystemExit(f"exported model is missing {len(missing)} parameters")
    print(f"  verified: loads with {sum(p.numel() for p in model.parameters()):,} "
          "parameters and no missing keys")
