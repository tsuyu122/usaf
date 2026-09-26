"""The importance scorer, and the accounting the OOM skip path needs.

The scorer silently used to drop batches that ran out of memory and then return
whatever it had, so a caller ranking on those scores could not tell them from a
complete pass - and a RigL reselection driven by a partial pass keeps a
different set of experts, quietly, forever after.
"""
import pytest
import torch
import torch.nn as nn

from usaf.importance import ImportanceScorer


class _Toy(nn.Module):
    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        # The scorer skips embed_tokens and lm_head by name, so a model made of
        # only those two has nothing left to score and backward has nothing to
        # differentiate. The block is the part the ranking is actually about.
        self.block = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids=None, labels=None, **kw):
        h = torch.tanh(self.block(self.embed_tokens(input_ids)))
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return type("Out", (), {"loss": loss, "logits": logits})()


def _loader(n=3, batch=2, seq=4, vocab=16):
    return [{"input_ids": torch.randint(0, vocab, (batch, seq)),
             "labels": torch.randint(0, vocab, (batch, seq))}
            for _ in range(n)]


def _scorer():
    return ImportanceScorer(_Toy(), torch.device("cpu"), torch.float32)


def test_a_clean_pass_scores_everything_it_asked_for():
    s = _scorer()
    scores = s.compute_scores(_loader(n=3))
    assert scores, "no scores at all"
    assert s.batches_scored == 3
    assert s.batches_skipped == 0


def test_max_batches_stops_the_pass_early():
    s = _scorer()
    s.compute_scores(_loader(n=6), max_batches=2)
    assert s.batches_scored == 2, s.batches_scored


def test_a_dropped_batch_is_counted_and_not_hidden():
    class _Flaky(_Toy):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids=None, labels=None, **kw):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
            return super().forward(input_ids=input_ids, labels=labels, **kw)

    s = ImportanceScorer(_Flaky(), torch.device("cpu"), torch.float32)
    scores = s.compute_scores(_loader(n=3))
    assert scores, "a dropped batch must not empty the scores"
    assert s.batches_skipped == 1, s.batches_skipped
    assert s.batches_scored == 2, s.batches_scored


def test_a_run_of_oom_gives_up_rather_than_spinning():
    class _Dead(_Toy):
        def forward(self, input_ids=None, labels=None, **kw):
            raise RuntimeError("CUDA out of memory")

    s = ImportanceScorer(_Dead(), torch.device("cpu"), torch.float32)
    with pytest.raises(RuntimeError, match="scored nothing"):
        s.compute_scores(_loader(n=30))


def test_an_error_that_is_not_about_memory_is_not_swallowed():
    class _Broken(_Toy):
        def forward(self, input_ids=None, labels=None, **kw):
            raise ValueError("the model is wrong")

    s = ImportanceScorer(_Broken(), torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError, match="the model is wrong"):
        s.compute_scores(_loader(n=2))


def test_the_embedding_is_left_out_of_the_ranking():
    s = _scorer()
    scores = s.compute_scores(_loader(n=2))
    assert not any("embed_tokens" in k for k in scores), sorted(scores)
    assert not any("lm_head" in k for k in scores), sorted(scores)


def test_scores_survive_a_save_and_load(tmp_path):
    s = _scorer()
    scores = s.compute_scores(_loader(n=2))
    p = tmp_path / "scores.pt"
    s.save_scores(scores, str(p))
    back = ImportanceScorer.load_scores(str(p))
    assert set(back) == set(scores)
    assert back and all(k in scores for k in back)
    for k in scores:
        # save_scores writes fp16, so the round trip is lossy by construction.
        # What has to hold is that the order survives, since the ranking is the
        # only thing these numbers are for - not the exact bits.
        assert back[k].dtype == torch.float16, (k, back[k].dtype)
        a = scores[k].reshape(-1)
        b = back[k].float().reshape(-1)
        assert a.shape == b.shape
        order = torch.argsort(a, descending=True)[:5]
        assert torch.equal(order, torch.argsort(b, descending=True)[:5]), k
        assert torch.allclose(b, a, rtol=1e-2, atol=1e-3), k

