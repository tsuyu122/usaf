import pytest

pytest.importorskip('transformers')


def test_the_cache_key_changes_when_the_forward_changes():
    # The cache holds hidden states the model's own forward produced. Anything
    # that changes what that forward does to an input has to be part of what
    # decides the cache is usable - and the key was a path, which does not
    # change when the arithmetic does.
    from usaf.train import cache_source_key

    class Plain:
        pass

    class Scaled:
        embedding_multiplier = 12.0

    class Other:
        embedding_multiplier = 1.0

    a = cache_source_key('/models/granite', Scaled())
    b = cache_source_key('/models/granite', Other())
    c = cache_source_key('/models/granite', Plain())

    assert a != b, (
        'a cache built with the multiplier and one built without it have the'
        'same key, so the wrong one is accepted')
    assert a != c, 'a scaled model and an unscaled one cannot share a cache'
    assert b == c, 'two stacks with no multiplier are the same cache'
    assert '/models/granite' in a, 'the path is still part of the key'


def test_the_cache_is_built_with_that_key():
    # The rule is useless if the call site still passes the bare path, which is
    # the same wiring half that let the embedding fix through once already.
    import inspect

    import usaf.train as t

    src = inspect.getsource(t._run_training)
    assert 'cache_source_key(config.model_path, model_cfg)' in src, (
        'the cache is keyed on the path alone')
