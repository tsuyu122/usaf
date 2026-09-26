"""Quantized expert streaming loader for MoE models (Qwen3-30B-A3B).

Loads a 4-bit quantized state dict from disk. Dequantizes expert weights on-the-fly
from CPU to GPU with an LRU cache. Integrates with forward hooks so only the active
expert layer consumes GPU memory.
"""

from __future__ import annotations

import fnmatch
import gc
import re
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .quantization import dequantize_4bit


def save_quantized_state_dict(
    q_dict: dict[str, Any],
    path: str,
    *,
    _use_new_zipfile_serialization: bool = True,
) -> None:
    """Save a quantized state dict to disk via ``torch.save``.

    Args:
        q_dict: Quantized state dict (output of ``quantize_state_dict``).
        path:   File path (e.g. ``"qwen3_a3b_q4.pt"``).
    """
    torch.save(q_dict, path, _use_new_zipfile_serialization=_use_new_zipfile_serialization)


def load_quantized_state_dict(
    path: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a quantized state dict from disk.

    Args:
        path:          File path to the quantized checkpoint.
        map_location:  Device to load tensors on (default "cpu").

    Returns:
        Quantized state dict — each quantized entry is a dict with
        keys "q", "s", "z", "shape".
    """
    loaded = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected quantized state dict (dict), got {type(loaded).__name__}")
    return loaded




def _apply_overlay(t: torch.Tensor, idx: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """Scatter compact trained values into a full-shape tensor at flat indices.

    ``t.reshape(-1)`` is not ``t.view(-1)``: on a non-contiguous tensor reshape
    returns a copy, and scattering into that copy changes nothing - the trained
    weights would be silently replaced by the pretrained ones and nothing would
    report it. The tensors on these paths are contiguous today, so the
    failure mode is latent rather than live, but it is one refactor away from
    being real. Making the copy explicit keeps the contract: the returned
    tensor is the one that carries the overlay.
    """
    if t.is_contiguous():
        flat = t.reshape(-1)
        flat.scatter_(0, idx.reshape(-1).to(torch.long), vals.detach().reshape(-1).to(flat.dtype))
        return t
    # a copia contigua e o que carrega o overlay, entao e ela que volta
    out = t.contiguous()
    out.reshape(-1).scatter_(0, idx.reshape(-1).to(torch.long), vals.detach().reshape(-1).to(out.dtype))
    return out
class QuantizedExpertCache:
    """LRU cache that holds quantized expert weights on CPU and dequantizes on demand.

    Each expert's weights stay in the packed 4-bit format on CPU until needed.
    When ``get_expert_weights`` is called, the weights are dequantized to fp16
    on the target device and cached.  An LRU policy evicts the least-recently-used
    expert when the cache is full.

    Lifecycle (integrated with ``setup_quantized_streaming``):
      1. pre_hook  → ``get_expert_weights(module_name)`` populates ``_parameters`` on GPU.
      2. forward   → expert module runs with dequantized fp16 weights.
      3. post_hook → weights are cleared from the module; GPU cache entry stays
         (or is evicted on next LRU miss).

    Args:
        quantized_dict:  Quantized state dict (output of
                         ``quantize_state_dict``).  Kept on CPU.
        device:          Target GPU device for dequantized weights.
        max_cached:      Maximum number of experts to keep in fp16 on GPU (LRU).
        group_size:      Group size used during quantization (must match
                         the value used in ``quantize_state_dict``).
    """

    def __init__(
        self,
        quantized_dict: dict[str, Any],
        device: torch.device,
        max_cached: int = 2,
        group_size: int = 128,
        expert_prefix: str = "model.layers.{i}.mlp.experts",
    ) -> None:
        self._q_dict = quantized_dict
        self._device = device
        self._max_cached = max(max_cached, 1)
        self._group_size = group_size
        # Expert module naming convention for this architecture, e.g.
        # "model.layers.{i}.mlp.experts". Inferred from the quantized keys when
        # possible so the cache does not silently address the wrong modules on a
        # layout that names its experts differently.
        self._expert_prefix = expert_prefix

        self._cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        # Pinned host copies, keyed by (expert module name, parameter name).
        # Bounded: a whole model walk would otherwise pin every expert once and
        # never release any of it.
        self._pinned: dict[tuple[str, str], torch.Tensor] = {}
        self._max_pinned: int = max(8, self._max_cached * 8)

        self.overlays: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        self._prefetched: dict[str, Future] = {}
        self._executor: ThreadPoolExecutor | None = None

        self._expert_to_params: dict[str, list[tuple[str, str]]] = {}
        self._build_index()

        self._resident: dict[str, dict[str, torch.Tensor]] = {}
        self._resident_active: bool = False

        self._vk_q4: dict[str, tuple] = {}
        self._vk_enabled: bool = False

    def _build_index(self) -> None:
        """Index quantized parameters by their expert module name.

        Scans ``_q_dict`` keys and groups them under the expert module
        they belong to.  For a key like ``model.layers.0.mlp.experts.gate_up_proj``,
        the expert module name is ``model.layers.0.mlp.experts`` and the local
        parameter name is ``gate_up_proj``.
        """
        expert_to_params: dict[str, list[tuple[str, str]]] = {}

        for full_name in self._q_dict:
            parts = full_name.rsplit(".", 1)
            if len(parts) != 2:
                continue
            module_name, param_name = parts
            expert_to_params.setdefault(module_name, []).append((full_name, param_name))

        self._expert_to_params = expert_to_params

        # Infer the expert module prefix from the actual keys when the caller
        # did not supply one explicitly. Every key has the shape
        # "<prefix with layer index>.<param>", so the common part across layers
        # identifies the convention without hardcoding ".mlp.experts".
        if expert_to_params:
            # Derive the convention from the keys themselves instead of
            # hardcoding ".mlp.experts". An expert module name is
            # "<stem><layer index><tail>"; replacing the index with {i} gives
            # the prefix, e.g. "model.layers.3.mlp.experts" ->
            # "model.layers.{i}.mlp.experts". Taking the stem from ONE module
            # and confirming the rest agree keeps this from being fooled by a
            # prefix that itself ends in a digit.
            pat = re.compile(r"^(?P<stem>.*?\.)(?P<idx>\d+)(?P<tail>\..*)?$")
            forms = set()
            for mod_name in expert_to_params:
                m = pat.match(mod_name)
                if m and m.group("stem"):
                    forms.add(m.group("stem") + "{i}" + (m.group("tail") or ""))
            if len(forms) == 1:
                self._expert_prefix = forms.pop()

    def _to_device(self, module_name: str,
                  cpu_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move a module's dequantized params to the device, pinned and async.

        Every call used to be a fresh t.to(device) per parameter, out of
        pageable host memory. A pageable source forces the driver to stage the
        data through an internal bounce buffer and the copy is synchronous: the
        host waits for it to land before it can issue anything else. This runs
        once per trainable layer per microbatch, and twice over per layer,
        because the backward replays a layer the forward already ran under
        no_grad.

        Pinning the host tensor the first time it is seen lets the DMA engine
        read it directly, with no staging, and is what makes non_blocking mean
        anything. The pinned buffer is kept so the cost is paid once per
        parameter rather than once per layer.

        The cache is keyed by module AND parameter name, never by the parameter
        name alone. Every layer of the stack has a "gate_up_proj" of exactly
        the same shape, so keying on the name returns whichever layer was pinned
        first and hands it to whichever layer asks next. The existing test for
        interleaved lookups fails on that with layer 3's weights in layer 0's
        hands, and it would have been a silent wrong-numbers bug on hardware
        where the shapes happen to agree, which is to say on every real model.

        What this deliberately does NOT do is copy into a reused device buffer.
        The same aliasing argument applies there, with worse consequences:
        overwriting the previous layer's weights in place, while a graph may
        still read them. A fresh allocation is the only version of this that
        cannot be wrong, and the fragmentation it causes is handled by the layer
        budget and the out-of-memory retry instead.

        A host that refuses to pin falls back to a plain copy rather than
        failing the run - only the speed is lost.
        """
        pinned = self._pinned
        dev = self._device
        out: dict[str, torch.Tensor] = {}
        for k, t in cpu_params.items():
            ckey = (module_name, k)
            src = pinned.get(ckey)
            if src is None or src.shape != t.shape or src.dtype != t.dtype:
                if dev.type == "cuda" and not t.is_pinned():
                    try:
                        t = t.pin_memory()
                    except RuntimeError:
                        pass
                pinned[ckey] = t
                src = t
                # Pinned host memory was growing once per expert for the whole
                # run and nothing ever released it. Pinned memory is unpageable
                # and unevictable, so a whole model walk ends up holding a full
                # second copy of the quantized weights that the host cannot
                # reclaim, on a host that also has to hold the checkpoint and
                # the model being trained. It is now bounded by the same LRU as
                # the GPU cache, and the key is the module plus the parameter
                # name, so evicting an expert releases both.
                while len(pinned) > self._max_pinned:
                    pinned.pop(next(iter(pinned)))
            out[k] = src.to(dev, non_blocking=True)
        return out
    def get_expert_weights(self, expert_module_name: str) -> dict[str, torch.nn.Parameter]:
        """Return dequantized fp16 Parameters for *expert_module_name*, on GPU.

        If the expert is already in the LRU cache it is promoted (move-to-end).
        Otherwise it is dequantized from the CPU-packed format and cached,
        evicting the least-recently-used expert if the cache is full.

        Args:
            expert_module_name: Full dotted module name, e.g.
                                ``"model.layers.0.mlp.experts"``.

        Returns:
            Dict mapping local parameter names (``"gate_up_proj"``, ``"down_proj"``)
            to ``nn.Parameter`` tensors on the target GPU device.
        """
        if expert_module_name in self._cache:
            self._cache.move_to_end(expert_module_name)
            cached = self._cache[expert_module_name]
            return {
                pname: torch.nn.Parameter(t, requires_grad=t.requires_grad)
                for pname, t in cached.items()
            }

        if expert_module_name in self._resident:
            cpu_params = self._get_resident(expert_module_name)
            # Dequantize ONLY the params that are not already resident. The
            # previous code ran a full 4-bit dequant of every param and then
            # discarded the result whenever the resident cache already had
            # it, so "dequant once, use forever" was false: evict_all() after
            # every forward kept the LRU cold and each forward re-paid the
            # dequant. It is only genuinely needed for params excluded from
            # the resident set (make_resident(only_params=[...])).
            missing = {
                ln for _, ln in self._expert_to_params.get(expert_module_name, [])
            } - set(cpu_params.keys())
            if missing:
                cpu_params.update(self._dequant_cpu(expert_module_name, only=missing))
            gpu_params = self._to_device(expert_module_name, cpu_params)
            self._cache[expert_module_name] = gpu_params
            return {
                pname: torch.nn.Parameter(t, requires_grad=True)
                for pname, t in gpu_params.items()
            }

        while len(self._cache) >= self._max_cached:
            self._cache.popitem(last=False)

        fut = self._prefetched.pop(expert_module_name, None)
        if fut is not None:
            cpu_params = fut.result()
        else:
            cpu_params = self._dequant_cpu(expert_module_name)

        gpu_params = self._to_device(expert_module_name, cpu_params)

        self._cache[expert_module_name] = gpu_params
        return {
            pname: torch.nn.Parameter(t, requires_grad=True)
            for pname, t in gpu_params.items()
        }

    def _dequant_cpu(self, expert_module_name: str,
                    only: set[str] | None = None) -> dict[str, torch.Tensor]:
        """Dequantize an expert module's params to CPU fp16 (overlay applied).

        Uses VK GPU dequant when _vk_q4_all has buffers for this param.

        Args:
            expert_module_name: Full dotted module name, e.g. "model.layers.0.mlp.experts".
            only: Optional set of *local* parameter names to dequantize. Any
                param not in the set is skipped, so a caller that already
                holds some params in fp16 (e.g. the resident cache) does not
                pay for a full 4-bit dequant of data it already has.
        """
        cpu_params: dict[str, torch.Tensor] = {}
        for full_name, local_name in self._expert_to_params.get(expert_module_name, []):
            if only is not None and local_name not in only:
                continue
            entry = self._q_dict.get(full_name)
            if entry is None:
                continue

            vk_info = self._vk_q4_all.get(full_name) if hasattr(self, '_vk_q4_all') else None
            if vk_info is not None:
                import usaf_vk
                q_h, s_h, z_h, out_rows, in_feats, total_elems = vk_info
                usaf_vk.dequant_pipe(q_h, s_h, z_h, self._vk_out_buf, out_rows, in_feats, 128)
                usaf_vk.barrier()
                raw = usaf_vk.download(self._vk_out_buf, [out_rows, in_feats])
                shape = entry[3] if isinstance(entry, tuple) else entry["shape"]
                t = torch.from_numpy(np.ascontiguousarray(raw.view(np.float16))).reshape(shape)
            elif isinstance(entry, dict) and "q" in entry:
                t = dequantize_4bit(
                    entry["q"], entry["s"], entry["z"], entry["shape"],
                    group_size=self._group_size,
                )
            elif isinstance(entry, tuple) and len(entry) == 4:
                t = dequantize_4bit(
                    entry[0], entry[1], entry[2], entry[3],
                    group_size=self._group_size,
                )
            elif isinstance(entry, torch.Tensor):
                t = entry.to(torch.float16)
            else:
                continue

            ov = self.overlays.get(full_name)
            if ov is not None:
                idx, vals = ov
                t = _apply_overlay(t, idx, vals)

            cpu_params[local_name] = t
        return cpu_params

    def _get_resident(self, expert_module_name: str) -> dict[str, torch.Tensor]:
        """Return resident fp16 params with overlay applied. Uses VK GPU dequant when enabled."""
        out: dict[str, torch.Tensor] = {}
        for local_name, base in self._resident[expert_module_name].items():
            full_name = f"{expert_module_name}.{local_name}"
            if self._vk_enabled and full_name in self._vk_q4:
                import usaf_vk
                q_h, s_h, z_h, out_rows, in_feats, total_elems = self._vk_q4[full_name]
                h_out = usaf_vk.create_buf(total_elems * 2, True)
                usaf_vk.dequant_pipe(q_h, s_h, z_h, h_out, out_rows, in_feats, 128)
                usaf_vk.barrier()
                raw = usaf_vk.download(h_out, [out_rows, in_feats])
                usaf_vk.destroy_buf(h_out)
                t = torch.from_numpy(np.ascontiguousarray(raw.view(np.float16))).reshape(base.shape)
            else:
                t = base.clone()
            ov = self.overlays.get(full_name)
            if ov is not None:
                idx, vals = ov
                t = _apply_overlay(t, idx, vals)
            out[local_name] = t
        return out

    def free_frozen(self, detach_at: int) -> None:
        """Free q4 entries for frozen layers 0..detach_at."""
        dropped = 0
        for li in range(detach_at + 1):
            prefix = f"model.layers.{li}."
            for key in list(self._q_dict.keys()):
                if key.startswith(prefix):
                    del self._q_dict[key]
                    dropped += 1
            expert_name = self._expert_prefix.format(i=li)
            self._expert_to_params.pop(expert_name, None)
            self._prefetched.pop(expert_name, None)
        gc.collect()

    @staticmethod
    def _available_host_bytes() -> int:
        """Host RAM that can be spent, leaving room for what is already held.

        The quantized payload, the checkpoint and the frozen activation cache
        are all in memory before this is called, so the number that matters is
        what is left, not what the machine has.
        """
        try:
            import os

            free = None
            if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
                free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            if free is None and os.name == "nt":
                import ctypes

                class _ME(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                me = _ME()
                me.dwLength = ctypes.sizeof(_ME)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(me)):
                    free = int(me.ullAvailPhys)
            if free is None:
                free = 8 * 1024 ** 3
            return int(free)
        except Exception:
            return 8 * 1024 ** 3

    def _resident_bytes(self, train_layers, only_params=None) -> int:
        """Bytes the dense resident set would need, without building it."""
        total = 0
        for li in train_layers:
            expert_name = self._expert_prefix.format(i=li)
            for full_name, local_name in self._expert_to_params.get(
                    expert_name, []):
                if only_params is not None and local_name not in only_params:
                    continue
                entry = self._q_dict.get(full_name)
                if entry is None:
                    continue
                if isinstance(entry, dict) and "shape" in entry:
                    shape = entry["shape"]
                elif isinstance(entry, (tuple, list)) and len(entry) == 4:
                    shape = entry[3]
                elif isinstance(entry, torch.Tensor):
                    shape = entry.shape
                else:
                    continue
                n = 1
                for s in shape:
                    n *= int(s)
                total += n * 2  # fp16
        return total

    def make_resident(self, train_layers, dequant_batch: int = 8, only_params=None) -> None:
        """Dequantize trainable-layer experts once to fp16 CPU RAM.

        Args:
            only_params: if set, only make these param names resident.
                         Others are streamed on-the-fly to save RAM.
        """
        self._resident_active = False
        # The resident set is the whole point: the trainable layers dequantized
        # once into host RAM instead of being streamed on every access. On a
        # model whose trainable layers are a large share of the experts that is
        # several gigabytes, held for the whole run, next to the checkpoint and
        # the quantized payload. Building it without asking whether it fits is
        # how a run dies on the host rather than on the card, after the card had
        # nothing to do with it. When it does not fit it is not built: the
        # streaming path is slower and correct, and the run continues.
        _wanted = sorted(train_layers)
        _need = self._resident_bytes(_wanted, only_params)
        _have = self._available_host_bytes()
        if _need > _have:
            short_gb = (_have - _need) / 1e9
            msg = (
                f"  resident set skipped: {_need / 1e9:.2f} GB of dense "
                f"experts for {len(_wanted)} layers, {short_gb:.2f} GB of "
                f"host RAM short. Experts stream from the quantized payload "
                f"instead, which is slower and fits."
            )
            print(msg, flush=True)
            return
        for li in _wanted:
            expert_name = self._expert_prefix.format(i=li)
            entries = self._expert_to_params.get(expert_name)
            if not entries:
                continue
            res: dict[str, torch.Tensor] = {}
            for full_name, local_name in entries:
                if only_params is not None and local_name not in only_params:
                    continue
                entry = self._q_dict.get(full_name)
                if entry is None:
                    continue
                if isinstance(entry, dict) and "q" in entry:
                    t = dequantize_4bit(
                        entry["q"], entry["s"], entry["z"], entry["shape"],
                        group_size=self._group_size,
                    )
                elif isinstance(entry, tuple) and len(entry) == 4:
                    t = dequantize_4bit(
                        entry[0], entry[1], entry[2], entry[3],
                        group_size=self._group_size,
                    )
                elif isinstance(entry, torch.Tensor):
                    t = entry.to(torch.float16)
                else:
                    continue
                res[local_name] = t
            self._resident[expert_name] = res
        self._resident_active = True

    def setup_vk_dequant(self, train_layers):
        """Upload q4 weights to Vulkan buffers for GPU dequant."""
        try:
            import os
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vulkan', 'build', 'Release'))
            os.add_dll_directory(os.environ.get('VULKAN_SDK', 'C:/VulkanSDK/1.4.341.1') + '/Bin')
            import usaf_vk

            from usaf.vulkan_spirv import configure_spirv
            configure_spirv(usaf_vk)
        except Exception as e:
            print(f"  [VK dequant] import failed: {e}", flush=True)
            return

        max_elems = 0
        n_uploaded = 0
        for li in sorted(train_layers):
            expert_name = self._expert_prefix.format(i=li)
            entries = self._expert_to_params.get(expert_name)
            if not entries:
                continue
            for full_name, local_name in entries:
                entry = self._q_dict.get(full_name)
                if entry is None:
                    continue
                q4 = entry[0].numpy()
                scales = entry[1].numpy()
                zeros = entry[2].numpy()
                shape = entry[3]
                in_feats = int(shape[-1])
                total_elems = 1
                for s in shape: total_elems *= int(s)
                out_rows = total_elems // in_feats
                n_groups = total_elems // 128

                q_gpu = np.ascontiguousarray(q4[:n_groups, :].reshape(out_rows, in_feats // 2))
                s_gpu = np.ascontiguousarray(scales[:n_groups].reshape(out_rows, in_feats // 128).astype(np.float16))
                z_gpu = np.ascontiguousarray(zeros[:n_groups].reshape(out_rows, in_feats // 128).astype(np.float16))

                h_q = usaf_vk.create_buf(q_gpu.nbytes, True)
                h_s = usaf_vk.create_buf(s_gpu.nbytes, True)
                h_z = usaf_vk.create_buf(z_gpu.nbytes, True)
                usaf_vk.upload(h_q, q_gpu)
                usaf_vk.upload(h_s, s_gpu)
                usaf_vk.upload(h_z, z_gpu)
                self._vk_q4[full_name] = (h_q, h_s, h_z, out_rows, in_feats, total_elems)
                max_elems = max(max_elems, total_elems)
                n_uploaded += 1

        if n_uploaded > 0:
            self._vk_enabled = True
        print(f"  [VK dequant] uploaded={n_uploaded} params, max_elems={max_elems}, enabled={self._vk_enabled}", flush=True)

    def setup_vk_streaming(self, max_layers: int = 999):
        """Upload q4 for layers 0..max_layers-1 to Vulkan. Accelerates _dequant_cpu.

        Args:
            max_layers: Only upload layers with index < max_layers (default: all)
        """
        try:
            import os
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vulkan', 'build', 'Release'))
            os.add_dll_directory(os.environ.get('VULKAN_SDK', 'C:/VulkanSDK/1.4.341.1') + '/Bin')
            import usaf_vk

            from usaf.vulkan_spirv import configure_spirv
            configure_spirv(usaf_vk)
        except Exception:
            return

        self._vk_q4_all = {}
        n = 0
        max_elems = 0
        for expert_name, entries in self._expert_to_params.items():
            parts = expert_name.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                try:
                    layer_idx = int(parts[2])
                    if layer_idx >= max_layers:
                        continue
                except ValueError:
                    pass
            for full_name, local_name in entries:
                entry = self._q_dict.get(full_name)
                if entry is None:
                    continue
                q4 = entry[0].numpy()
                scales = entry[1].numpy()
                zeros = entry[2].numpy()
                shape = entry[3]
                in_feats = int(shape[-1])
                total_elems = 1
                for s in shape:
                    total_elems *= int(s)
                out_rows = total_elems // in_feats
                n_groups = total_elems // 128

                q_gpu = np.ascontiguousarray(q4[:n_groups, :].reshape(out_rows, in_feats // 2))
                s_gpu = np.ascontiguousarray(scales[:n_groups].reshape(out_rows, in_feats // 128).astype(np.float16))
                z_gpu = np.ascontiguousarray(zeros[:n_groups].reshape(out_rows, in_feats // 128).astype(np.float16))

                h_q = usaf_vk.create_buf(q_gpu.nbytes, True)
                h_s = usaf_vk.create_buf(s_gpu.nbytes, True)
                h_z = usaf_vk.create_buf(z_gpu.nbytes, True)
                usaf_vk.upload(h_q, q_gpu)
                usaf_vk.upload(h_s, s_gpu)
                usaf_vk.upload(h_z, z_gpu)
                self._vk_q4_all[full_name] = (h_q, h_s, h_z, out_rows, in_feats, total_elems)
                max_elems = max(max_elems, total_elems)
                n += 1
        if max_elems > 0:
            self._vk_out_buf = usaf_vk.create_buf(max_elems * 2, True)
        print(f"  [VK streaming] uploaded={n} params, out_buf={max_elems*2/1e6:.0f}MB", flush=True)

    def apply_resident_overlays(self, active_idx, masters):
        """Scatter initial master values into resident tensors (call once after selection)."""
        for fname, aidx in active_idx.items():
            parts = fname.rsplit(".", 1)
            if len(parts) != 2:
                continue
            expert_name, local_name = parts
            if expert_name not in self._resident or local_name not in self._resident[expert_name]:
                continue
            base = self._resident[expert_name][local_name]
            aidx_flat = aidx.reshape(-1).to(torch.long)
            vals = masters[fname].data.detach().to(torch.float16)
            base.reshape(-1).scatter_(0, aidx_flat, vals)

    def sync_resident(self, active_idx, masters):
        """Propagate optimizer master updates into resident fp16 tensors.

        Call after each SparseAdam.step() — the masters are updated, so we
        scatter their new values back into the resident base (in-place).
        """
        if not self._resident_active:
            return
        for fname, aidx in active_idx.items():
            parts = fname.rsplit(".", 1)
            if len(parts) != 2:
                continue
            expert_name, local_name = parts
            if expert_name not in self._resident or local_name not in self._resident[expert_name]:
                continue
            base = self._resident[expert_name][local_name]
            aidx_flat = aidx.reshape(-1).to(torch.long)
            vals = masters[fname].data.detach().to(torch.float16)
            base.reshape(-1).scatter_(0, aidx_flat, vals)

    def prefetch(self, expert_module_name: str) -> None:
        """Schedule async CPU dequantization of an expert module."""
        if getattr(self, '_prefetch_disabled', False):
            return
        if (
            expert_module_name in self._cache
            or expert_module_name in self._prefetched
            or expert_module_name not in self._expert_to_params
        ):
            return
        if expert_module_name in self._resident:
            res = self._resident[expert_module_name]
            deq_entries = self._expert_to_params.get(expert_module_name, [])
            if all(ln in res for _, ln in deq_entries):
                return
        if (
            expert_module_name in self._cache
            or expert_module_name in self._prefetched
            or expert_module_name not in self._expert_to_params
            or expert_module_name in self._resident
        ):
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        self._prefetched[expert_module_name] = self._executor.submit(
            self._dequant_cpu, expert_module_name
        )

    def clear_prefetch(self) -> None:
        """Drop pending prefetches, waiting on in-flight jobs."""
        for f in self._prefetched.values():
            f.cancel()
        for f in self._prefetched.values():
            if not f.cancelled():
                try:
                    f.result(timeout=60)
                except Exception:
                    pass
        self._prefetched.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def evict(self, expert_module_name: str) -> None:
        """Explicitly remove an expert from the GPU cache."""
        self._cache.pop(expert_module_name, None)

    def evict_all(self) -> None:
        """Clear all GPU-cached experts (pending prefetches are kept)."""
        self._cache.clear()

    @property
    def cached_experts(self) -> list[str]:
        """Names of experts currently cached on GPU (most recent first)."""
        return list(reversed(self._cache.keys()))

    @property
    def device(self) -> torch.device:
        """Target GPU device."""
        return self._device

    def estimate_vram_mb(self) -> float:
        """Estimate GPU VRAM used by the current cache in MiB."""
        total = 0
        for params in self._cache.values():
            for t in params.values():
                total += t.numel() * t.element_size()
        return total / (1024 * 1024)

    def estimate_cpu_mb(self) -> float:
        """Estimate CPU RAM used by the quantized storage in MiB."""
        total = 0
        for entry in self._q_dict.values():
            if isinstance(entry, dict) and "q" in entry:
                total += entry["q"].numel() * entry["q"].element_size()
                total += entry["s"].numel() * entry["s"].element_size()
                total += entry["z"].numel() * entry["z"].element_size()
            elif isinstance(entry, torch.Tensor):
                total += entry.numel() * entry.element_size()
        return total / (1024 * 1024)


class SparseGradStore:
    """Captures only the active elements of each expert tensor for sparse gradient tracking.

    Precomputes per-expert local flat indices of active elements and their
    positions in a compact per-tensor gradient vector aligned with ``active_idx``.
    Used via the ``module._grad_capture = (store, prefix)`` protocol.
    """

    def __init__(
        self,
        active_idx: dict[str, torch.Tensor],
        shapes: dict[str, Any],
    ) -> None:
        self.compact: dict[str, torch.Tensor] = {}
        self._maps: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        self._dev_idx: dict[tuple[str, int], torch.Tensor] = {}

        for name, idx in active_idx.items():
            idx = idx.reshape(-1).to(device="cpu", dtype=torch.long)
            self.compact[name] = torch.zeros(idx.numel(), dtype=torch.float32)
            shape = shapes[name]
            slice_size = 1
            for s in shape[1:]:
                slice_size *= int(s)
            e_of = torch.div(idx, slice_size, rounding_mode="floor")
            per_expert: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for ei in torch.unique(e_of).tolist():
                ei = int(ei)
                pos = (e_of == ei).nonzero(as_tuple=False).reshape(-1)
                local = idx[pos] - ei * slice_size
                per_expert[ei] = (local, pos)
            self._maps[name] = per_expert

    def zero_(self) -> None:
        for v in self.compact.values():
            v.zero_()

    def add(self, full_name: str, ei: int, g: torch.Tensor) -> None:
        per_expert = self._maps.get(full_name)
        if not per_expert:
            return
        entry = per_expert.get(int(ei))
        if entry is None:
            return
        local, pos = entry
        key = (full_name, int(ei))
        lidx = self._dev_idx.get(key)
        if lidx is None or lidx.device != g.device:
            lidx = local.to(g.device)
            self._dev_idx[key] = lidx
        vals = g.detach().reshape(-1).gather(0, lidx).to("cpu").float()
        self.compact[full_name].index_add_(0, pos, vals)

    def n_captured(self) -> int:
        return len(self.compact)


class TopKImportanceStore:
    """Importance scoring without dense full-shape grad buffers.

    Keeps top ``frac * cand_mult`` |grad| candidates per expert slice at capture
    time, then merges per tensor with a global top-k in ``select``.
    Same ``add`` protocol as SparseGradStore.
    """

    def __init__(self, shapes: dict[str, Any], frac: float, cand_mult: float = 2.0) -> None:
        self._shapes = shapes
        self._frac = frac
        self._cand = cand_mult
        self._vals: dict[str, dict[int, torch.Tensor]] = {n: {} for n in shapes}
        self._idx: dict[str, dict[int, torch.Tensor]] = {n: {} for n in shapes}

    def zero_(self) -> None:
        for n in self._vals:
            self._vals[n].clear()
            self._idx[n].clear()

    def add(self, full_name: str, ei: int, g: torch.Tensor) -> None:
        if full_name not in self._vals:
            return
        flat = g.detach().to("cpu").float().reshape(-1).abs()
        k = max(1, int(flat.numel() * self._frac * self._cand))
        v, i = torch.topk(flat, min(k, flat.numel()))
        self._vals[full_name][int(ei)] = v
        self._idx[full_name][int(ei)] = i.to(torch.long) + int(ei) * flat.numel()

    def select(self, frac: float | None = None) -> dict[str, torch.Tensor]:
        """Merge per-expert candidates into global flat active indices."""
        frac = self._frac if frac is None else frac
        out: dict[str, torch.Tensor] = {}
        for name, shape in self._shapes.items():
            if not self._vals[name]:
                continue
            vals = torch.cat(list(self._vals[name].values()))
            idxs = torch.cat(list(self._idx[name].values()))
            numel = 1
            for s in shape:
                numel *= int(s)
            k = min(max(1, int(numel * frac)), vals.numel())
            top = torch.topk(vals, k).indices
            out[name] = idxs[top]
        return out

    def n_captured(self) -> int:
        return sum(len(d) for d in self._vals.values())


def _move_module_tensors_to(module: nn.Module, target: torch.device) -> None:
    """Move all params and buffers of a single module to ``target`` (DML-safe)."""
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


def setup_quantized_streaming(
    model: nn.Module,
    quantized_dict: dict[str, Any],
    device: torch.device,
    *,
    max_cached_experts: int = 2,
    group_size: int = 128,
    expert_pattern: str | None = None,
    verbose: bool = True,
) -> nn.Module:
    """Stream quantized expert weights from CPU to GPU during forward passes.

    Expert weights are stored in packed 4-bit format on CPU and dequantized
    on-the-fly via ``QuantizedExpertCache`` when pre-hooks fire. Non-expert
    parameters stay GPU-resident.

    Args:
        model:              OLMoE / Qwen3-MoE model.
        quantized_dict:     State dict quantized via ``quantize_state_dict``.
        device:             Target GPU device.
        max_cached_experts: LRU cache size.
        group_size:         Group size used during quantization.
        expert_pattern:     Glob matching expert module names. None (the
                            default) derives it from the quantized keys, so a
                            non-standard expert layout still matches.
        verbose:            Print summary statistics.

    Returns:
        Model with streaming hooks installed.
    """
    # Build the cache first so the expert naming convention is inferred from
    # the quantized keys, then derive the matching glob from that prefix.
    # The previous hardcoded default "*.mlp.experts" matched nothing on any
    # architecture that names its experts differently, so no hooks were
    # installed and the experts silently ran on their (cleared) meta tensors.
    cache = QuantizedExpertCache(
        quantized_dict,
        device=device,
        max_cached=max_cached_experts,
        group_size=group_size,
    )
    if expert_pattern is None:
        expert_pattern = cache._expert_prefix.format(i="*")

    torch.device("cpu")

    n_gpu = n_cpu = 0
    bytes_gpu = bytes_cpu = 0

    expert_module_names: list[str] = []
    for mname, mod in model.named_modules():
        is_expert = fnmatch.fnmatch(mname, expert_pattern)
        if is_expert:
            expert_module_names.append(mname)
            mod._parameters.clear()
            continue

        for pname, p in list(mod._parameters.items()):
            if p is None:
                continue
            nbytes = p.numel() * p.element_size()
            n_gpu += 1
            bytes_gpu += nbytes
        _move_module_tensors_to(mod, device)

    bytes_cpu = cache.estimate_cpu_mb() * 1024 * 1024
    n_cpu = len(quantized_dict)

    if verbose:
        print(f"  GPU (residente): {n_gpu} tensores, {bytes_gpu / 1e9:.2f}GB")
        print(f"  CPU (4-bit):      {n_cpu} entradas, ~{bytes_cpu / 1e9:.2f}GB")
        print(f"  Cache LRU:        {max_cached_experts} experts max")

    state: dict[str, int] = {"resident": 0, "max": 0}

    model._expert_grads: dict[str, torch.Tensor] = {}
    grad_store = model._expert_grads

    def make_pre_hook(mod_name: str):
        def pre_hook(module: nn.Module, _args: list) -> None:
            weights = cache.get_expert_weights(mod_name)
            for pname, param in weights.items():
                module._parameters[pname] = param
                if param.requires_grad:
                    full = f"{mod_name}.{pname}"

                    def _cap(p: nn.Parameter, _full: str = full) -> None:
                        if p.grad is not None:
                            grad_store[_full] = p.grad.detach()

                    param.register_post_accumulate_grad_hook(_cap)
            state["resident"] += 1
            state["max"] = max(state["max"], state["resident"])
            return None

        return pre_hook

    def make_post_hook():
        def post_hook(module: nn.Module, _args: list, output: torch.Tensor) -> torch.Tensor:
            module._parameters.clear()
            state["resident"] -= 1
            return output

        return post_hook

    model._stream_state = state

    expert_modules: list[nn.Module] = []
    for name, mod in model.named_modules():
        if name in expert_module_names:
            expert_modules.append(mod)
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

    model._quantized_cache = cache
    return model


def get_quantized_cache(model: nn.Module) -> QuantizedExpertCache | None:
    """Retrieve the ``QuantizedExpertCache`` attached to a model by
    ``setup_quantized_streaming``."""
    return getattr(model, "_quantized_cache", None)


def apply_captured_expert_grads(
    model: nn.Module,
    scale: float = 1.0,
) -> int:
    """Copy captured expert grads to CPU-resident canonical Parameters.

    Expert module parameters are ephemeral (cleared after each forward).
    Gradients captured via ``post_accumulate_grad_hook`` are transferred back.
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
            p.grad = gc.to(device=p.device, dtype=p.dtype)
            applied += 1

    store.clear()
    return applied


def load_and_stream(
    model: nn.Module,
    quantized_path: str,
    device: torch.device,
    **stream_kwargs: Any,
) -> nn.Module:
    """Convenience: load quantized weights from disk and install streaming hooks.

    Args:
        model:           OLMoE / Qwen3-MoE model.
        quantized_path:  Path to the ``.pt`` file produced by
                         ``save_quantized_state_dict``.
        device:          Target GPU device.
        **stream_kwargs: Passed to ``setup_quantized_streaming``.

    Returns:
        Model with streaming hooks installed.
    """
    q_dict = load_quantized_state_dict(quantized_path)
    return setup_quantized_streaming(model, q_dict, device, **stream_kwargs)
