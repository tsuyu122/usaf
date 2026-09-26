"""
The reselect merge has to mean the same thing at scale as it does on the
tiny fixture, and it has to do it without the memory.

The original was set arithmetic in Python: sets of millions of Python ints,
intersected, differenced and sorted. It is correct - and it costs tens of
bytes per element in a set of boxed ints, times two sets and two sorted lists,
per tensor, on a 110M-parameter active set. The replacement is unique plus
two isin calls on tensors, which is the same operation without boxing.

Correctness is checked against the set version directly, on sizes big enough
that the set version's cost is visible, so a divergence in ordering or
duplication cannot hide behind a fixture too small to notice.
"""

import torch


def merge_python(old, nw):
    old_set = set(old.reshape(-1).tolist())
    nw_set = set(nw.reshape(-1).tolist())
    kept = sorted(old_set & nw_set)
    candidates = sorted(nw_set - old_set)
    fill = max(0, len(old_set) - len(kept))
    return torch.tensor(kept + candidates[:fill], dtype=torch.long)


def merge_tensor(old, nw):
    _old_uniq = old.reshape(-1).to(torch.long).unique()
    _nw_uniq = nw.reshape(-1).to(torch.long).unique()
    _is_kept = torch.isin(_nw_uniq, _old_uniq)
    kept_idx = _nw_uniq[_is_kept]
    grown_idx = _nw_uniq[~_is_kept]
    fill = max(0, _old_uniq.numel() - kept_idx.numel())
    return torch.cat([kept_idx, grown_idx[:fill]])


def _rand_idx(n, total, gen):
    return torch.randperm(total, generator=gen)[:n].sort().values


def test_the_two_merges_agree():
    gen = torch.Generator().manual_seed(0)
    total = 200_000
    for trial in range(5):
        old = _rand_idx(20_000 + trial * 3_000, total, gen)
        nw = _rand_idx(20_000 + trial * 1_500, total, gen)
        a = merge_python(old, nw)
        b = merge_tensor(old, nw)
        assert a.numel() == b.numel(), (
            f"""trial {trial}: {a.numel()} vs {b.numel()}"""
        )
        assert torch.equal(a.sort().values, b.sort().values), (
            f"""trial {trial}: different index sets"""
        )


def test_the_merge_keeps_the_full_budget():
    """The point of the fill: a drop has to be replaced, or the active set
    shrinks on every reselection and the run converges to nothing."""
    gen = torch.Generator().manual_seed(1)
    total = 100_000
    old = _rand_idx(30_000, total, gen)
    # Deliberately disjoint from old, so nothing is kept.
    nw = _rand_idx(30_000, total, gen).add(total)
    out = merge_tensor(old, nw)
    assert out.numel() == old.numel(), (
        f"""budget fell from {old.numel()} to {out.numel()}"""
    )


def test_duplicate_input_indices_do_not_inflate_the_set():
    """new_idx comes from a topk over concatenated candidates; duplicates are
    possible, and a set silently collapsed them."""
    old = torch.arange(0, 200, 2, dtype=torch.long)
    nw = torch.cat([old[:50], old[:50]]).sort().values
    a = merge_python(old, nw)
    b = merge_tensor(old, nw)
    assert torch.equal(a.sort().values, b.sort().values), (a, b)
    assert b.numel() == min(old.numel(), nw.unique().numel()), b.numel()
