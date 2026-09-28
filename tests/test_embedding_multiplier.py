import inspect

import pytest
import torch

pytest.importorskip('transformers')


def test_the_embedding_scale_is_read_from_the_config():
    # 1.0 has to be a real answer, for the stacks with no multiplier - which is
    # every stack this repository has run until now.
    from usaf.train import embedding_scale

    class Cfg:
        pass

    c = Cfg()
    assert embedding_scale(c) == 1.0, 'no multiplier means none'
    c.embedding_multiplier = 12.0
    assert embedding_scale(c) == 12.0
    c.embedding_multiplier = None
    assert embedding_scale(c) == 1.0, 'a None multiplier is not a multiplier'
    c.embedding_multiplier = 0.0
    assert embedding_scale(c) == 1.0, 'zero is a missing value, not a scale'


def test_scaling_the_embedding_actually_scales_it():
    # The rule and the code that applies it are different things, and the rule
    # on its own has already let a broken fix through once.
    from usaf.train import scaled_embedding

    class Cfg:
        embedding_multiplier = 12.0

    emb = torch.nn.Embedding(32, 64)
    ids = torch.tensor([[1, 5, 9, 2]])

    got = scaled_embedding(emb, Cfg())(ids)
    bare = emb(ids)

    ratio = float(got.norm() / bare.norm())
    assert abs(ratio - 12.0) < 1e-3, f'got {ratio:.3f}x, expected 12x'

    class Plain:
        pass

    plain = scaled_embedding(emb, Plain())(ids)
    assert float(plain.norm()) == pytest.approx(float(bare.norm())), (
        'a stack with no multiplier has to come through untouched')


def test_the_replay_is_wired_to_the_scaling_embedding():
    # This is the half that was missing. Reverting the call site inside
    # _prelude left the rule correct and the run broken, and the two tests
    # above still passed, because neither of them reads the caller.
    import usaf.train as t

    src = inspect.getsource(t._run_training)
    assert 'scaled_embedding(embed, model_cfg)' in src, (
        'the prelude does not go through scaled_embedding, so the multiplier'
        'is not applied and every layer sees an input worth 12 times less'
        'than the one the model was trained on')
    assert 'hidden = embedded(input_ids)' in src, (
        'the prelude calls something other than the scaled embedding')
