"""Numerical validation of the remaining Vulkan kernels against PyTorch.

Only the attention kernel had a numerical test. rmsnorm, rope, gemm and the
4-bit dequantizer are the other three the training path actually calls, and each
is a place where a wrong constant or a transposed index produces plausible but
wrong activations. These pin them against a PyTorch reference.

The extension takes and returns raw fp16 bit patterns (uint16 views), which is
what the stub signature says; pass a float16 view instead and the kernel
reinterprets the bits as a denormal float.

They are skipped unless the usaf_vk extension has been built and a Vulkan
device is available.
"""
import numpy as np
import pytest

from tests.test_vulkan_kernels import _load_vk, _spirv_dir


def _bits(a):
    """float16/float32 -> the uint16 bit pattern the extension expects."""
    return np.ascontiguousarray(a, dtype=np.float16).view(np.uint16)


def _f32(a):
    """uint16 bit pattern from the extension -> float32."""
    return np.ascontiguousarray(a, dtype=np.uint16).view(np.float16).astype(np.float32)


@pytest.fixture(scope="module")
def vk():
    m = _load_vk()
    _spirv_dir(m)
    return m


def test_rmsnorm_matches_torch(vk):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    rows, cols = 7, 128
    x = rng.standard_normal((rows, cols)).astype(np.float16)
    w = (rng.standard_normal(cols) + 1.0).astype(np.float16)
    got = _f32(vk.rmsnorm(_bits(x), _bits(w), rows, cols))

    tx = torch.from_numpy(x).float()
    tw = torch.from_numpy(w).float()
    ref = (tx / torch.sqrt((tx ** 2).mean(-1, keepdim=True) + 1e-6)) * tw
    scale = max(float(ref.abs().max()), 1e-6)
    err = float(np.abs(got - ref.numpy()).max()) / scale
    assert got.shape == (rows, cols), got.shape
    assert err < 1e-2, f"rmsnorm relative max error {err}"


def test_gemm_matches_torch(vk):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    m_dim, k_dim, n_dim = 5, 64, 48
    a = (rng.standard_normal((m_dim, k_dim)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((k_dim, n_dim)) * 0.3).astype(np.float16)
    got = _f32(vk.gemm(_bits(a), _bits(b), m_dim, k_dim, n_dim))
    ref = (torch.from_numpy(a).float() @ torch.from_numpy(b).float()).numpy()
    scale = max(float(np.abs(ref).max()), 1e-6)
    err = float(np.abs(got - ref).max()) / scale
    assert got.shape == (m_dim, n_dim), got.shape
    assert err < 1e-2, f"gemm relative max error {err}"


def test_rope_is_a_pure_rotation(vk):
    """Rotary embedding must rotate, never scale, each head dimension pair."""
    batch, n_heads, n_kv, seq, head_dim = 1, 4, 2, 16, 64
    rng = np.random.default_rng(2)
    q = (rng.standard_normal((batch, n_heads, seq, head_dim)) * 0.5).astype(np.float16)
    k = (rng.standard_normal((batch, n_kv, seq, head_dim)) * 0.5).astype(np.float16)
    # The kernel follows the HuggingFace half-split format: cos and sin have
    # shape [seq, head_dim] with duplicated halves (cos[i] == cos[i + hd/2]).
    # The pair (x[i], x[i + hd/2]) is rotated, so cos**2 + sin**2 must be 1 or
    # the transform scales the norm. A 45 degree turn gives 1/sqrt(2) for both.
    inv_sqrt2 = np.float16(1.0 / np.sqrt(2.0))
    cos = np.full((seq, head_dim), inv_sqrt2, dtype=np.float16)
    sin = np.full((seq, head_dim), inv_sqrt2, dtype=np.float16)
    out_q, out_k = vk.rope(
        _bits(q), _bits(k), _bits(cos), _bits(sin),
        batch, n_heads, n_kv, seq, head_dim,
    )
    out_q = _f32(out_q)
    out_k = _f32(out_k)

    assert out_q.shape == (batch, n_heads, seq, head_dim), out_q.shape
    assert out_k.shape == (batch, n_kv, seq, head_dim), out_k.shape

    n_in = np.linalg.norm(q.astype(np.float32).reshape(batch, n_heads, seq, head_dim), axis=-1)
    n_out = np.linalg.norm(out_q.reshape(batch, n_heads, seq, head_dim), axis=-1)
    rel = np.abs(n_out - n_in).max() / max(float(n_in.max()), 1e-6)
    assert rel < 1e-2, f"rope changed the norm by {rel}, it must be a rotation"

    x_full = q.astype(np.float32).reshape(batch, n_heads, seq, head_dim)
    lo = x_full[:, :, :, : head_dim // 2]
    hi = x_full[:, :, :, head_dim // 2:]
    got_lo = out_q.reshape(batch, n_heads, seq, head_dim)[:, :, :, : head_dim // 2]
    got_hi = out_q.reshape(batch, n_heads, seq, head_dim)[:, :, :, head_dim // 2:]
    # out[i] = x[i]*cos - x[i+hd/2]*sin, out[i+hd/2] = x[i+hd/2]*cos + x[i]*sin
    c = float(inv_sqrt2)
    expect_lo = lo * c - hi * c
    expect_hi = hi * c + lo * c
    scale = max(float(np.abs(lo).max()), 1e-6)
    err = max(
        float(np.abs(got_lo - expect_lo).max()),
        float(np.abs(got_hi - expect_hi).max()),
    ) / scale
    assert err < 1e-2, f"rope rotation error {err}"


def test_dequant_q4_pipeline_matches_torch(vk):
    """The 4-bit dequantizer must agree with the PyTorch dequant."""
    torch = pytest.importorskip("torch")
    usaf_q = pytest.importorskip("usaf.quantization")

    torch.manual_seed(0)
    rows, cols = 4, 128
    x = torch.randn(rows, cols).half()
    q, s, z, shape = usaf_q.quantize_4bit(x, group_size=128)
    ref = usaf_q.dequantize_4bit(q, s, z, shape, group_size=128).float().numpy()

    # q holds two packed 4-bit values per byte; it must go up as raw bytes.
    # Converting it to float first silently destroys the nibbles.
    packed = np.ascontiguousarray(q.numpy().view(np.uint8))
    hq = vk.create_buf(packed.nbytes)
    hs = vk.create_buf(s.numel() * 2)
    hz = vk.create_buf(z.numel() * 2)
    ho = vk.create_buf(rows * cols * 2)
    try:
        vk.upload(hq, packed)
        vk.upload(hs, _bits(s.float().numpy()))
        vk.upload(hz, _bits(z.float().numpy()))
        vk.dequant_pipe(hq, hs, hz, ho, rows, cols, 128)
        got = _f32(vk.download(ho, (rows, cols)))
    finally:
        for h in (hq, hs, hz, ho):
            vk.destroy_buf(h)

    assert got.shape == (rows, cols), got.shape
    scale = max(float(np.abs(ref).max()), 1e-6)
    err = float(np.abs(got - ref).max()) / scale
    assert err < 1e-2, f"dequant relative max error {err}"
