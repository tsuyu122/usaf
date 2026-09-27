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


def test_the_resident_set_outlives_the_importance_pass():
    # The frozen cache's resident set must still be alive when the importance
    # pass runs, and gone after it.
    #
    # It used to be cleared right after the build, which is before. The
    # importance pass replays every layer by calling the module directly, and
    # that call relies on the pre-hook having populated the expert weights from
    # the resident set. With it gone the module has no gate_up_proj and the run
    # dies with AttributeError - after an hour of cache building, on ZAYA, with
    # a log full of healthy progress lines and no hint of what went wrong.
    src = _train_source()
    assert "cache.make_resident(_cache_layers)" in src, (
        "the cache layers are never made resident, so the build streams instead"
    )
    i_make = src.find("cache.make_resident(_cache_layers)")
    i_free = src.find("cache.free_frozen(DETACH_AT)")
    assert 0 <= i_make < i_free, (
        "it is freed before it is built, which does nothing at all"
    )
    i_clear = src.find("cache._resident.clear()")
    i_imp = src.find("imp_store = TopKImportanceStore")
    assert 0 <= i_imp < i_clear, (
        "the resident set is cleared before the importance pass reads it; "
        "the module loses its expert weights and raises AttributeError"
    )


def test_the_resident_set_is_released_only_after_the_importance_pass():
    # ...and it is still released. Keeping 11.7 GB for the whole run is how the
    # trainable layers' own resident set gets skipped for want of memory.
    src = _train_source()
    assert src.count("cache._resident.clear()") == 1, (
        "cleared zero times it is leaked; more than once is a second release "
        "point that the importance pass has already run through"
    )
    i_imp = src.find("imp_store = TopKImportanceStore")
    i_clear = src.find("cache._resident.clear()")
    assert i_clear > i_imp > 0, (i_clear, i_imp)
    i_active = src.find("cache._resident_active = False")
    assert 0 <= i_active <= i_clear + 200, (
        "the flag is left True while the set is empty, so the next caller "
        "believes a resident set exists and skips rebuilding it"
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
