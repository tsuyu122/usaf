import gc

import pytest
import torch


class _Tally:
    """Counts collected parts. A set of indices would cap at one per attempt."""

    def __init__(self):
        self.freed = 0

    def bump(self):
        self.freed += 1


class _Part:
    def __init__(self, tally):
        self._tally = tally

    def __del__(self):
        self._tally.bump()


def _run(train_layers, tally, oom_above, stash=None):
    """The retry loop as main() runs it, over a stand-in loader."""
    seen = []
    free_at_start = []

    def _load_and_run(_layers):
        # Counted BEFORE this attempt allocates: if the previous attempt's
        # model were still reachable, the retry stacked a second copy on top of
        # the first - the failure this whole path exists to prevent.
        free_at_start.append(tally.freed == 2 * len(seen))
        seen.append(len(_layers))
        model = [_Part(tally), _Part(tally)]
        if stash is not None:
            stash.append(model)  # the shape of a real leak
        if len(_layers) > oom_above:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate")
        return model

    _layers_now = list(train_layers)
    out = None
    for _attempt in range(1, 6):
        try:
            out = _load_and_run(_layers_now)
            break
        except torch.OutOfMemoryError as _oom:
            _why = str(_oom).splitlines()[0] if _oom.args else ""
            gc.collect()
            if len(_layers_now) <= 1:
                raise SystemExit(f"no fit: {_why}")
            _keep = max(1, int(len(_layers_now) * 0.75))
            _layers_now = _layers_now[-_keep:]
    return out, seen, _layers_now, free_at_start


def test_the_retry_shrinks_the_layer_count_until_it_fits():
    _out, seen, final, _f = _run([0, 1, 2, 3, 4, 5, 6, 7], _Tally(), oom_above=2)
    assert seen == [8, 6, 4, 3, 2], seen
    assert final == [6, 7], final


def test_each_retry_starts_with_the_previous_model_already_freed():
    _out, _seen, _final, free = _run([0, 1, 2, 3, 4, 5, 6, 7], _Tally(), oom_above=2)
    assert free == [True] * 5, free


def test_the_liveness_check_would_notice_a_retry_that_leaks():
    # The check has to be able to fail, or it pins nothing. A stash is the
    # shape of a real leak: something outside the attempt frame holding the
    # model, so the retry does stack a second copy on top of the first.
    stash = []
    _out, _seen, _final, free = _run([0, 1, 2, 3, 4, 5, 6, 7], _Tally(),
                                     oom_above=2, stash=stash)
    assert free[0] is True
    assert any(not f for f in free[1:]), free


def test_one_layer_reports_instead_of_looping_forever():
    with pytest.raises(SystemExit) as exc:
        _run([0, 1], _Tally(), oom_above=0)
    assert "no fit" in str(exc.value)

