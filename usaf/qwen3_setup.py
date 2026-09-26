"""Setup compartilhado do Qwen3-30B-A3B com streaming quantizado (DML).

Extraído de train_qwen3_12h.py para reuso em scripts de avaliação
(eval_bugfix_score.py). O caminho de treino mantém sua própria cópia
validada — este helper é para inferência/scoring.
"""
import os

import torch
from safetensors import safe_open
from transformers import AutoConfig

from .moe_loader import QuantizedExpertCache
from .qwen3moe_dml import patch_qwen3moe_for_dml
from .train import materialize_meta_tensors
from .utils import get_dml_device


def load_qwen3_streaming(src: str, q4_dir: str, max_cached: int = 1):
    """Monta o modelo em meta device + streaming quantizado. Retorna (model, cache, device)."""
    patch_qwen3moe_for_dml()
    import torch_directml_native
    torch_directml_native.disable_tiled_resources(True)
    device = get_dml_device()

    with torch.device("meta"):
        cfg = AutoConfig.from_pretrained(src)
        from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
        model = Qwen3MoeForCausalLM(cfg)

    st_files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    wf = {}
    for fn in st_files:
        with safe_open(os.path.join(src, fn), framework="pt") as sf:
            for key in sf.keys():
                wf[key] = fn

    mp = dict(model.named_parameters())
    for name in sorted(wf.keys()):
        if ".mlp.experts." in name or name not in mp:
            continue
        with safe_open(os.path.join(src, wf[name]), framework="pt") as sf:
            tensor = sf.get_tensor(name).half()
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        obj._parameters[parts[-1]] = torch.nn.Parameter(tensor.to(device), requires_grad=False)

    # The RoPE frequencies used to be rebuilt here with
    # getattr(mod, "dim", getattr(mod, "head_dim", 128)) to guess the rotary
    # dimension, and zero-filled everything else. Rotary modules expose neither
    # attribute, so the guess always landed on 128: a model whose real head_dim
    # differs produced frequencies of the wrong length and the first forward
    # died with "The size of tensor a (8) must match tensor b (128)". The same
    # bug existed a second time here while the copy in train.py had already been
    # fixed, which is what a duplicated fix always risks. It is one function now
    # and both call sites share it.
    materialize_meta_tensors(model, cfg, device, src, expert_modules=set())

    q_dict = torch.load(os.path.join(q4_dir, "experts_q4.pt"), map_location="cpu", weights_only=True)
    cache = QuantizedExpertCache(q_dict, device, max_cached=max_cached, group_size=128)

    for mname, mod in model.named_modules():
        if mname.endswith(".mlp.experts"):
            mod._parameters.clear()
            if hasattr(mod, "_buffers"):
                mod._buffers.clear()

    for mname, mod in model.named_modules():
        if not mname.endswith(".mlp.experts"):
            continue

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

    model.eval()
    return model, cache, device


def apply_checkpoint_overlays(cache, ckpt_path: str) -> int:
    """Aplica os masters compactos de um checkpoint de treino como overlays."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    n = 0
    for fname, aidx in ckpt["active_idx"].items():
        aidx = aidx.reshape(-1).to(torch.long)
        vals = torch.nn.Parameter(ckpt["masters"][fname].float(), requires_grad=False)
        cache.overlays[fname] = (aidx, vals)
        n += 1
    return n
