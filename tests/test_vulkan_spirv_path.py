"""The shaders must be found where they actually are.

Four call sites used to force <package>/vulkan/build/spirv, an in-tree build
directory that exists in no installed copy. Setting it overrode the extension's
own module-relative lookup - which had already been fixed - with a path that was
never there, so USE_VK=1 died with "Cannot open SPIR-V" after the loader itself
was correct.
"""
import os

from usaf.vulkan_spirv import configure_spirv, spirv_dir_for


class _FakeVk:
    __file__ = None

    def __init__(self):
        self.set_to = []

    def set_spirv_path(self, d):
        self.set_to.append(d)


def test_a_directory_without_the_shaders_is_not_configured(tmp_path):
    """Pointing at an existing-but-empty directory is just as broken as
    pointing at a missing one, and harder to notice."""
    fake = _FakeVk()
    fake.__file__ = str(tmp_path / "usaf_vk.pyd")
    os.makedirs(tmp_path / "spirv")

    assert spirv_dir_for(fake) is None
    assert configure_spirv(fake) is None
    assert fake.set_to == [], "set the shader directory to an empty folder"


def test_shaders_next_to_the_extension_are_used(tmp_path):
    fake = _FakeVk()
    fake.__file__ = str(tmp_path / "usaf_vk.pyd")
    spirv = tmp_path / "spirv"
    spirv.mkdir()
    (spirv / "rmsnorm_fp16.spv").write_bytes(b"x")

    assert spirv_dir_for(fake) == str(spirv)
    configure_spirv(fake)
    assert fake.set_to == [str(spirv)]


def test_a_partial_shader_set_is_rejected(tmp_path):
    """Having the directory is not enough; the kernels have to be in it."""
    fake = _FakeVk()
    fake.__file__ = str(tmp_path / "usaf_vk.pyd")
    spirv = tmp_path / "spirv"
    spirv.mkdir()
    (spirv / "something_else.spv").write_bytes(b"x")

    assert spirv_dir_for(fake) is None


def test_no_path_is_forced_when_the_module_has_no_file():
    """Some loaders expose no __file__; the module-relative default is then the
    only thing left, so nothing should be forced."""
    fake = _FakeVk()
    assert fake.__file__ is None
    assert configure_spirv(fake) is None
    assert fake.set_to == []


def test_no_module_sets_a_stale_in_tree_path():
    """The regression: nothing may hand the extension a build/ path that the
    loader would resolve better on its own."""
    import glob
    import re

    root = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.join(os.path.dirname(root), "usaf")
    pattern = re.compile(r"set_spirv_path\s*\([^)]*build")
    offenders = []
    for path in glob.glob(os.path.join(pkg, "*.py")):
        with open(path, encoding="utf-8") as f:
            if pattern.search(f.read()):
                offenders.append(os.path.basename(path))
    assert not offenders, f"these still force an in-tree shader path: {offenders}"
