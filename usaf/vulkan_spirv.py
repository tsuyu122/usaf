"""Locate the compiled SPIR-V shaders.

The C++ extension resolves each kernel relative to its own loaded module when
no directory has been set explicitly (see resolve_spirv in pybind_module.cpp),
which is where CMake stages them next to the built .pyd.

The Python side used to call set_spirv_path with <package>/vulkan/build/spirv,
an in-tree build directory that does not exist in any installed copy. Setting it
overrode the correct module-relative default with a path that was never there,
so USE_VK=1 failed with:

    RuntimeError: Cannot open SPIR-V: .../usaf/vulkan/build/spirv/rmsnorm_fp16.spv

after the loader had already been fixed, because four call sites still forced
the stale path.
"""
from __future__ import annotations

import os


def spirv_dir_for(usaf_vk) -> str | None:
    """Return a directory that actually holds the shaders, or None.

    None means "leave the extension's own module-relative resolution alone",
    which is both correct and the only option that survives the package being
    copied somewhere without its build tree.
    """
    candidates = []

    # Where a built extension keeps them: next to the .pyd.
    mod_file = getattr(usaf_vk, "__file__", None)
    if mod_file:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(mod_file)), "spirv"))

    # In-tree build, for someone who built it where the sources are.
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "vulkan", "build", "spirv"))
    candidates.append(os.path.join(here, "vulkan", "build", "Release", "spirv"))

    for d in candidates:
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "rmsnorm_fp16.spv")):
            return d
    return None


def configure_spirv(usaf_vk) -> str | None:
    """Point the extension at the shaders, if we can find them.

    Returns the directory actually configured, or None when the module-relative
    default is in play.
    """
    d = spirv_dir_for(usaf_vk)
    if d is not None:
        usaf_vk.set_spirv_path(d)
    return d
