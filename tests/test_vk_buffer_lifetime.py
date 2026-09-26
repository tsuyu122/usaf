"""A Vulkan buffer must not be destroyed twice.

forward_full listed the buffers to free by name, and hq_rope_in is an alias of
hq_normed - and hk_rope_in of hk_normed - whenever the model has q_norm and
k_norm. That is every Qwen3, which is the family train.py drives. Listing both
names called destroy_buf twice on the same handle.

Destroying a Vulkan buffer twice is undefined behaviour. The driver may
complain, or the handle may already have been handed back out to something
else, in which case this frees a buffer that is in use. These buffers are
created and destroyed per forward rather than drawn from a pool, so nothing
catches the mismatch before the damage.

The duble below returns zeros of the right shape and records what happened to
what. The numbers are meaningless on purpose: this is about lifetime, and a
reference implementation would only make the test fail for arithmetic reasons.
"""
import numpy as np
import pytest

from usaf import vk_layer


class FakeVk:
    """Records buffer lifetime. Every op records that it wrote something."""

    def __init__(self):
        self.nbytes = {}
        self.destroyed = []
        self._next = 1

    def create_buf(self, nbytes, host_visible=True):
        h = self._next
        self._next += 1
        self.nbytes[h] = int(nbytes)
        return h

    def upload(self, h, arr):
        self.nbytes[h] = max(self.nbytes[h], int(arr.nbytes))

    def download(self, h, shape):
        # The caller immediately does .view(np.float16) on whatever comes back,
        # so this has to be fp16 already or the element count halves.
        return np.zeros(tuple(shape), dtype=np.float16)

    def destroy_buf(self, h):
        self.destroyed.append(h)

    def barrier(self):
        pass

    def _touch(self, h, n):
        self.nbytes[h] = max(self.nbytes.get(h, 0), int(n))

    def gemm_pipe(self, a, b, out, m, k, n):
        self._touch(out, m * n * 2)

    def rmsnorm_pipe(self, x, w, out, rows, cols, eps):
        self._touch(out, rows * cols * 2)

    def residual_add_pipe(self, a, b, out, n):
        self._touch(out, n * 2)

    # The parameter names have to match the extension's C API.
    def rope_pipe(self, qi, ki, cos, sin, qo, ko, B, nH, nKV, S, hd):  # noqa: N803
        self._touch(qo, B * S * nH * hd * 2)
        self._touch(ko, B * S * nKV * hd * 2)


def _weights(hidden, n_heads, n_kv_heads, head_dim, with_norm):
    w = {
        "input_layernorm.weight": np.ones(hidden, dtype=np.float32),
        "post_attention_layernorm.weight": np.ones(hidden, dtype=np.float32),
        "self_attn.q_proj.weight": np.zeros((hidden, n_heads * head_dim), dtype=np.float32),
        "self_attn.k_proj.weight": np.zeros((hidden, n_kv_heads * head_dim), dtype=np.float32),
        "self_attn.v_proj.weight": np.zeros((hidden, n_kv_heads * head_dim), dtype=np.float32),
        "self_attn.o_proj.weight": np.zeros((n_heads * head_dim, hidden), dtype=np.float32),
    }
    if with_norm:
        w["self_attn.q_norm.weight"] = np.ones(head_dim, dtype=np.float32)
        w["self_attn.k_norm.weight"] = np.ones(head_dim, dtype=np.float32)
    return w


def _run(fake, with_norm):
    hidden, n_heads, n_kv_heads, head_dim = 16, 4, 2, 4
    batch, seq = 1, 8
    layer = vk_layer.VKLayer(hidden, n_heads, n_kv_heads, head_dim,
                            _weights(hidden, n_heads, n_kv_heads, head_dim,
                                     with_norm))
    cos = np.zeros((seq, head_dim), dtype=np.float32)
    layer.forward_full(np.zeros((batch, seq, hidden), dtype=np.float32),
                       cos, cos)


@pytest.mark.parametrize("with_norm", [True, False],
                         ids=["qwen3_has_q_norm", "no_q_norm"])
def test_no_buffer_is_destroyed_twice(monkeypatch, with_norm):
    fake = FakeVk()
    monkeypatch.setattr(vk_layer, "usaf_vk", fake, raising=False)
    monkeypatch.setattr(vk_layer, "HAS_VK", True, raising=False)
    _run(fake, with_norm)
    seen = set()
    dupes = []
    for h in fake.destroyed:
        if h in seen:
            dupes.append(h)
        seen.add(h)
    assert not dupes, f"destroy_buf called twice on {dupes}. Destroying a "f"Vulkan buffer twice is undefined, and the handle may already belong to "f"something else."


