"""Padding is a thing the model should not be trained to produce.

The labels of a short chunk were padded with pad_token_id, but the trainer
counts labels != -100 to decide what the loss covers and the evaluator counts
the same to report a perplexity. Padding written into the labels as a real
token id is therefore not ignored - it is a target, and the loss counts it.

Every source file shorter than the context window was training the model to
emit padding, and the reported loss was diluted by positions that carry no
information. In a C++ corpus most files are under 2048 tokens, so this was the
common case and not the edge one.
"""
import pytest

from usaf.data import tokenize_text


class _Tok:
    pad_token_id = 0

    def __init__(self, n):
        self.n = n

    def __call__(self, text, **kw):
        return {"input_ids": list(range(1, self.n + 1))}


def _chunks(text_len, ctx=16, overlap=4):
    return list(tokenize_text("x" * 1000, _Tok(text_len),
                              context_length=ctx, overlap=overlap))


def test_a_short_text_pads_the_labels_with_ignore_not_with_pad():
    [only] = _chunks(5, ctx=16)
    assert only["input_ids"] == [1, 2, 3, 4, 5] + [0] * 11, only["input_ids"]
    assert only["labels"] == [1, 2, 3, 4, 5] + [-100] * 11, only["labels"]
    assert 0 not in only["labels"], "the padding is still a real id"


def test_no_position_that_is_only_padding_counts_toward_the_loss():
    [only] = _chunks(5, ctx=16)
    counted = sum(1 for v in only["labels"] if v != -100)
    assert counted == 5, f"{counted} positions counted, there are 5 real ones"


def test_a_full_chunk_has_no_ignored_positions():
    chunks = _chunks(40, ctx=16, overlap=4)
    assert len(chunks) > 1
    for c in chunks:
        assert -100 not in c["labels"], c
        assert len(c["input_ids"]) == 16


def test_the_input_ids_and_the_labels_are_not_the_same_object():
    [only] = _chunks(5, ctx=16)
    assert only["input_ids"] is not only["labels"]


def test_an_overlap_wider_than_the_window_is_refused():
    # stride becomes zero and range() rejects a zero step with a message about
    # the argument rather than about the overlap that caused it.
    with pytest.raises(ValueError, match="overlap"):
        list(tokenize_text("x", _Tok(100), context_length=8, overlap=8))
