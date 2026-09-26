"""Numerical validation of the Vulkan attention kernel against PyTorch.

These tests are skipped unless the usaf_vk extension module has been built
and a Vulkan device is available, so they never break a CPU-only checkout.

Build the extension with:
    cmake -S usaf/vulkan -B build -DCMAKE_BUILD_TYPE=Release \\
          -DPython3_EXECUTABLE=$(which python)
    cmake --build build --config Release --target usaf_vk

The attention kernel previously issued barrier() inside non-uniform control
flow ("for d = tid; d < hd; d += 256"), which is undefined behaviour in
GLSL/Vulkan and produced a max abs error of 1.99 against this reference.
"""
import os

import numpy as np
import pytest
import torch


def _load_vk():
    try:
        import usaf_vk
    except ImportError:
        pytest.skip("usaf_vk extension not built")
    return usaf_vk


def _spirv_dir(usaf_vk=None):
    # The compiled shaders sit next to the build tree that produced the
    # extension module, which is not necessarily inside the repository - an
    # out-of-tree build puts it elsewhere entirely. Look there first, then
    # fall back to the in-tree layout.
    candidates = []
    if usaf_vk is not None:
        mod_dir = os.path.dirname(os.path.abspath(usaf_vk.__file__))
        candidates.append(mod_dir)
        candidates.append(os.path.dirname(mod_dir))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        here,
        os.path.dirname(here),
        os.path.dirname(os.path.dirname(here)),
    ]
    env = os.environ.get("USAF_VK_SPIRV_DIR")
    if env:
        candidates.insert(0, env)
    for cand in candidates:
        spv = os.path.join(cand, "spirv")
        if os.path.isfile(os.path.join(spv, "attention.spv")):
            return spv
    pytest.skip(
        "compiled SPIR-V not found (build the usaf_vk target, or point "
        "USAF_VK_SPIRV_DIR at the directory containing it)"
    )


def _attention(usaf_vk, scores, v, nH, nKV, S, hd, causal):
    usaf_vk.set_spirv_dir("") if hasattr(usaf_vk, "set_spirv_dir") else None
    n_out = nH * S * hd
    bs = usaf_vk.create_buf(scores.numel() * 2, True)
    bv = usaf_vk.create_buf(v.numel() * 2, True)
    bo = usaf_vk.create_buf(n_out * 2, True)
    try:
        usaf_vk.upload(bs, scores.numpy())
        usaf_vk.upload(bv, v.numpy())
        usaf_vk.attn_softmax_pipe(bs, bv, bo, nH, nKV, S, hd, causal)
        usaf_vk.barrier()
        raw = np.asarray(usaf_vk.download(bo, [n_out]))
    finally:
        usaf_vk.destroy_buf(bs)
        usaf_vk.destroy_buf(bv)
        usaf_vk.destroy_buf(bo)
    return raw.view(np.float16).reshape(nH, S, hd).astype(np.float32)


def _reference(scores, v, nH, nKV, S, hd, causal):
    lg = scores.float()
    if causal:
        # -1e30 rather than -inf: -inf propagates NaN through the softmax and
        # makes the comparison meaningless.
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        lg = lg.masked_fill(mask, -1e30)
    rep = v.float().repeat_interleave(nH // nKV, dim=0)
    return torch.softmax(lg, dim=-1) @ rep


CASES = [
    # (nH, nKV, S, hd, causal)
    (4, 2, 64, 32, 0),
    (4, 2, 64, 32, 1),
    (8, 8, 128, 64, 1),
    (2, 1, 256, 128, 0),
    (1, 1, 64, 128, 1),
    (8, 2, 128, 64, 1),
]


@pytest.mark.parametrize("nH,nKV,S,hd,causal", CASES)
def test_attention_kernel_matches_torch(nH, nKV, S, hd, causal):
    usaf_vk = _load_vk()
    usaf_vk.set_spirv_path(_spirv_dir(usaf_vk))
    torch.manual_seed(0)
    scores = torch.randn(nH, S, S).half()
    v = torch.randn(nKV, S, hd).half()
    out = _attention(usaf_vk, scores, v, nH, nKV, S, hd, causal)
    ref = _reference(scores, v, nH, nKV, S, hd, causal).numpy()
    err = float(np.abs(out - ref).max())
    # fp16 storage of the inputs/outputs bounds the achievable error.
    assert err < 5e-3, (
        f"attention kernel max abs error {err:.5f} (nH={nH} nKV={nKV} "
        f"S={S} hd={hd} causal={causal}); the kernel has drifted from the "
        f"PyTorch reference"
    )
