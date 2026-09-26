"""A decoder stack whose layers are not all alike.

USAF's forward assumed one shared rotary, a 4-D mask tensor and a bare
hidden state out of every layer. A stack marked with config.layer_types breaks
all three at once - it keeps one RoPE per layer type and looks the frequency
buffer up by name, attention wants a dict of named masks, and the layer returns
a (hidden, router_summary) pair.

The gap was invisible until a real run: ZAYA1-8B downloaded, was detected, was
quantized, and then died in the first forward with
AttributeError: 'ZayaRotaryEmbedding' object has no attribute 'None_inv_freq'.
Nothing before that point says the architecture is unsupported.

These tests use fakes shaped like the real ZAYA rotary and layer so the
adapter's contract is pinned without needing the 17 GB model.
"""

import torch

from usaf.train import _DecoderAdapter


class Cfg:
    def __init__(self, layer_types=None):
        self.layer_types = layer_types


class ZayaLikeRotary:
    """Looks its buffer up by layer type, exactly as ZayaRotaryEmbedding does."""

    def __init__(self, n_freq=4):
        # Keyed by the full buffer name, the way ZayaRotaryEmbedding registers
        # them - the layer type is only part of that name.
        self.bufs = {
            'default_inv_freq': torch.arange(n_freq).float() + 1.0,
            'hybrid_sliding_inv_freq': torch.arange(n_freq).float() * 10.0 + 100.0,
        }
        self.calls = []

    def __call__(self, x, position_ids=None, layer_type=None):
        self.calls.append(layer_type)
        key = f'{layer_type}_inv_freq'
        if key not in self.bufs:
            raise AttributeError(
                f"'ZayaLikeRotary' object has no attribute '{key}'"
            )
        f = self.bufs[key]
        return (x[..., :1] + f).to(x.dtype), (x[..., :1] - f).to(x.dtype)


class PlainRotary:
    """The ordinary case: one buffer, no layer_type argument."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, position_ids=None):
        self.calls.append('plain')
        return (x[..., :1] + 1.0).to(x.dtype), (x[..., :1] - 1.0).to(x.dtype)


class PairLayer:
    """Records what it was handed and returns the pair ZAYA returns."""

    def __init__(self):
        self.seen = None

    def __call__(self, h, attention_mask=None, position_ids=None,
                 position_embeddings=None):
        self.seen = {
            'mask': attention_mask,
            'pe': position_embeddings,
        }
        return h + 1.0, None


class TensorLayer:
    def __init__(self):
        self.seen = None

    def __call__(self, h, attention_mask=None, position_ids=None,
                 position_embeddings=None):
        self.seen = {'mask': attention_mask, 'pe': position_embeddings}
        return h + 1.0


def _run(cfg, rotary, layers, h, pos):
    mask = torch.zeros(1, 1, h.shape[1], h.shape[1])
    ad = _DecoderAdapter(layers, rotary, cfg, h, pos, mask)
    for i in range(len(layers)):
        h = ad.call(i, h, pos)
    return ad, h


def test_a_plain_stack_is_untouched():
    h = torch.zeros(1, 3, 2)
    pos = torch.arange(3).unsqueeze(0)
    layers = [TensorLayer() for _ in range(3)]
    rot = PlainRotary()
    ad, out = _run(Cfg(None), rot, layers, h, pos)
    assert not ad.hybrid
    assert rot.calls == ['plain'], rot.calls
    assert isinstance(ad.mask, torch.Tensor), type(ad.mask)
    for lyr in layers:
        assert isinstance(lyr.seen['mask'], torch.Tensor)
        assert isinstance(lyr.seen['pe'], tuple)
    assert torch.equal(out, torch.full((1, 3, 2), 3.0))


def test_each_layer_gets_the_rope_of_its_own_type():
    h = torch.zeros(1, 3, 2)
    pos = torch.arange(3).unsqueeze(0)
    types = ['default', 'hybrid_sliding', 'default', 'hybrid_sliding']
    layers = [PairLayer() for _ in types]
    rot = ZayaLikeRotary()
    ad, _ = _run(Cfg(types), rot, layers, h, pos)
    assert ad.hybrid
    # One rotary call per distinct type, not one per layer.
    assert rot.calls == ['default', 'hybrid_sliding'], rot.calls
    for i, lyr in enumerate(layers):
        want = ad.pe[types[i]]
        assert torch.equal(lyr.seen['pe'][0], want[0]), i
        assert torch.equal(lyr.seen['pe'][1], want[1]), i


def test_the_rope_actually_differs_between_types():
    """Otherwise the per-layer plumbing passes while proving nothing."""
    h = torch.zeros(1, 3, 2)
    pos = torch.arange(3).unsqueeze(0)
    ad = _DecoderAdapter([PairLayer()], ZayaLikeRotary(), Cfg(['default', 'hybrid_sliding']),
                         h, pos, torch.zeros(1, 1, 3, 3))
    a, b = ad.pe['default'], ad.pe['hybrid_sliding']
    assert not torch.equal(a[0], b[0])
    assert float(a[0].max()) < float(b[0].max())


def test_a_heterogeneous_stack_gets_a_named_mask():
    h = torch.zeros(1, 3, 2)
    pos = torch.arange(3).unsqueeze(0)
    mask4d = torch.zeros(1, 1, 3, 3)
    ad = _DecoderAdapter([PairLayer()], ZayaLikeRotary(), Cfg(['default']),
                         h, pos, mask4d)
    assert isinstance(ad.mask, dict), type(ad.mask)
    # ZayaAttention reads mask_mapping.get('causal') and .get('padding').
    assert torch.equal(ad.mask['causal'], mask4d)
    assert ad.mask['padding'] is None


def test_a_pair_return_is_unwrapped_to_the_hidden_state():
    h = torch.zeros(1, 3, 2)
    pos = torch.arange(3).unsqueeze(0)
    ad, out = _run(Cfg(['default']), ZayaLikeRotary(), [PairLayer()], h, pos)
    assert isinstance(out, torch.Tensor), type(out)
    assert out.shape == h.shape
    assert not torch.equal(out, h)


def test_a_pair_return_keeps_the_stack_usable():
    """The summary slot must not be threaded into the next layer as hidden."""
    h = torch.zeros(1, 3, 2)
    pos = torch.arange(3).unsqueeze(0)
    layers = [PairLayer() for _ in range(3)]
    ad, out = _run(Cfg(['default'] * 3), ZayaLikeRotary(), layers, h, pos)
    assert out.shape == (1, 3, 2)
    assert torch.equal(out, torch.full((1, 3, 2), 3.0))


def test_a_missing_rope_type_is_reported():
    h = torch.zeros(1, 3, 2)
    pos = torch.arange(3).unsqueeze(0)
    try:
        _DecoderAdapter([PairLayer()], ZayaLikeRotary(), Cfg(['no_such_type']),
                         h, pos, torch.zeros(1, 1, 3, 3))
    except AttributeError as e:
        assert 'no_such_type_inv_freq' in str(e), e
    else:
        raise AssertionError('an unknown layer type should not be accepted')

