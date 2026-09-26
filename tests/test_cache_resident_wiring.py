"""
The frozen cache re-runs its layers once per sample.

On ZAYA that is 124 samples over layers 0..28, and the experts those layers
need are 4-bit in the payload. Streaming them means 11.7 GB of
dequantization per sample; the resident set means it once. The build spent
about 32 s per sample on a real T4 against a 4.93 ms matmul layer, so the
dequant is the whole cost and the matmul is noise.

Two things have to hold and neither is visible from the outside: the
resident set is actually built for the frozen layers, and it is actually
released afterwards. Holding both it and the trainable layers resident is
how make_resident's budget check skips the second one, which is a slowdown
that looks like a hardware limit.
"""

def _train_source() -> str:
    with open(
        r"""C:\Users\hm\Projects\usaf\usaf\train.py""", encoding="""utf-8"""
    ) as f:
        return f.read()


def test_the_cache_layers_are_built_resident():
    src = _train_source()
    assert "cache.make_resident(_cache_layers)" in src, (
        "the frozen layers are still streamed once per sample"
    )


def test_the_resident_set_is_released_after_the_build():
    src = _train_source()
    assert "cache.free_frozen(DETACH_AT)" in src, (
        "11.7 GB of frozen experts stays resident for the whole run"
    )
    i_make = src.find("cache.make_resident(_cache_layers)")
    i_free = src.find("cache.free_frozen(DETACH_AT)")
    assert 0 <= i_make < i_free, (
        "it is freed before it is built, which does nothing at all"
    )


def test_the_trainable_resident_call_is_still_there():
    src = _train_source()
    assert "cache.make_resident(train_layers)" in src, (
        "the trainable layers lost their resident set"
    )


def test_only_frozen_layers_are_made_resident_for_the_cache():
    src = _train_source()
    i_filter = src.find("if i <= DETACH_AT")
    assert i_filter != -1, (
        "nothing bounds the cache resident set to the frozen layers, so it "
        "would cover all 40 and skip the trainable one"
    )
