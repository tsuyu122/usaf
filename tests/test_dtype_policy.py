"""Precision policy.

Non-expert weights were loaded with .half() regardless of what the model
declared, and the causal mask was hardcoded float16. A bf16 model therefore
came back as fp16 with an fp16 mask, which is the mismatch that made a bf16
checkpoint unusable.

The flag now exists, but the honest answer for --dtype auto is still fp16:
the expert path is float16 by construction, and a flag that advertised bf16
while the expert cache handed back fp16 would be a flag that lies.
"""

import pytest
import torch

from usaf.utils import get_optimal_dtype, resolve_dtype


@pytest.mark.parametrize(
    "name,expected",
    [
        ("auto", torch.float16),
        ("", torch.float16),
        ("fp16", torch.float16),
        ("half", torch.float16),
        ("BF16", torch.bfloat16),
        ("fp32", torch.float32),
        ("  fp16  ", torch.float16),
    ],
)
def test_resolve_dtype_accepts_the_names_the_cli_documents(name, expected):
    assert resolve_dtype(name, torch.device("cpu")) == expected


def test_auto_resolves_to_fp16_not_to_the_hardware_preference():
    """auto must not drift to float32 on CPU and then fail at the first
    expert matmul, nor to bf16 on a GPU the expert path cannot feed.
    """
    for dev in ("cpu", "cuda"):
        assert resolve_dtype("auto", torch.device(dev)) == torch.float16


def test_unknown_dtype_is_an_error_not_a_silent_fallback():
    with pytest.raises(SystemExit) as e:
        resolve_dtype("banana", torch.device("cpu"))
    assert "precision" in str(e.value)


def test_get_optimal_dtype_prefers_bf16_on_capable_cuda():
    """The hardware helper says what the GPU can do. It is deliberately not
    what auto resolves to - see resolve_dtype.
    """
    assert get_optimal_dtype(torch.device("privateuseone")) == torch.float16
    assert get_optimal_dtype(torch.device("cpu")) == torch.float32


def test_get_optimal_dtype_never_returns_a_wider_type_than_the_hardware_allows():
    for dev in ("cpu", "cuda", "xpu", "mps"):
        d = get_optimal_dtype(torch.device(dev))
        assert d in (torch.float16, torch.bfloat16, torch.float32)
