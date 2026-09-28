import inspect

import pytest
import torch

pytest.importorskip('transformers')


class Cfg:
    embedding_multiplier = 12.0
    logits_scaling = 6.0


def test_the_two_scalars_are_read_from_the_config():
    # Both are scalars GraniteMoe applies outside the module that owns them, and
    # both were missing from the replay. Missing has to be a real answer: every
    # other stack this repository has run has neither, and must not be touched.
    from usaf.train import embedding_scale, logits_scaling

    class Plain:
        pass

    p = Plain()
    assert embedding_scale(p) == 1.0
    assert logits_scaling(p) == 1.0

    c = Cfg()
    assert embedding_scale(c) == 12.0
    assert logits_scaling(c) == 6.0

    c.embedding_multiplier = None
    assert embedding_scale(c) == 1.0
    c.embedding_multiplier = 0.0
    assert embedding_scale(c) == 1.0
    c.logits_scaling = 0.0
    assert logits_scaling(c) == 1.0


def test_scaling_the_embedding_actually_scales_it():
    from usaf.train import scaled_embedding

    emb = torch.nn.Embedding(32, 64)
    ids = torch.tensor([[1, 5, 9, 2]])
    bare = emb(ids)

    ratio = float(scaled_embedding(emb, Cfg())(ids).norm() / bare.norm())
    assert abs(ratio - 12.0) < 1e-3, f'got {ratio:.3f}x, expected 12x'

    class Plain:
        pass

    plain = scaled_embedding(emb, Plain())(ids)
    assert float(plain.norm()) == pytest.approx(float(bare.norm()))


def test_the_logits_are_divided_like_the_model_divides_them():
    from usaf.train import scaled_logits

    head = torch.nn.Linear(64, 200, bias=False)
    h = torch.randn(1, 5, 64)
    bare = head(h)

    got = scaled_logits(head, Cfg(), h, None)
    ratio = float(bare.norm() / got.norm())
    assert abs(ratio - 6.0) < 1e-3, f'logits are {ratio:.3f}x, expected 6x'

    class Plain:
        pass

    plain = scaled_logits(head, Plain(), h, None)
    assert torch.allclose(plain, bare)


def test_the_training_replay_is_wired_to_both():
    # Reverting either call site leaves the rules correct and the run broken, and
    # the tests above still pass, because none of them reads the caller. That is
    # the eighth time that has happened in this repository.
    import usaf.train as t

    src = inspect.getsource(t._run_training)
    assert 'scaled_embedding(embed, model_cfg)' in src, 'prelude not scaled'
    assert 'hidden = embedded(input_ids)' in src, 'prelude not the scaled embed'
    assert 'scaled_logits(lm_head, model_cfg, hidden, norm_fn)' in src, 'loss not scaled'
