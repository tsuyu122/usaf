"""A dataset that was asked for and not loaded must not be quietly replaced.

get_eval_texts falls through to the synthetic C++ corpus whenever the path is
missing or absent, and returns it under the name that was requested. The CLI has
no way to pass a path at all, so `eval_cli --datasets jsonl` measures the C++
corpus, files the result under "jsonl", and prints a perplexity that belongs to
nothing the caller asked for.

That is the shape that matters here: a number that is plausible, labelled
correctly, and wrong. Comparing a base model against a fine-tuned one through
it compares two runs of the same surrogate dataset and the difference means
nothing.
"""
import json

import pytest

from usaf.eval.datasets import MAX_SNIPPET_CHARS, get_eval_texts


def test_the_synthetic_corpus_is_itself_fine(tmp_path):
    got = get_eval_texts("synthetic-cpp", None, 3)
    assert len(got) == 3


def test_a_jsonl_that_exists_is_read(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"text": "primeira"}) + chr(10)
                 + json.dumps({"text": "segunda"}) + chr(10),
                 encoding="utf-8")
    got = get_eval_texts("jsonl", str(p), 10)
    assert got == ["primeira", "segunda"], got


def test_a_jsonl_path_that_does_not_exist_is_refused(tmp_path):
    missing = str(tmp_path / "nao_existe.jsonl")
    with pytest.raises(ValueError, match="nao_existe.jsonl"):
        get_eval_texts("jsonl", missing, 10)


def test_asking_for_jsonl_with_no_path_is_refused():
    """The CLI can only reach this state, and it used to be silent."""
    with pytest.raises(ValueError, match="path"):
        get_eval_texts("jsonl", None, 10)


def test_an_unknown_dataset_name_is_refused():
    with pytest.raises(ValueError, match="meu-dataset"):
        get_eval_texts("meu-dataset", None, 10)


def test_long_texts_are_truncated_to_the_snippet_limit(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("x" * (MAX_SNIPPET_CHARS + 500), encoding="utf-8")
    got = get_eval_texts("text", str(p), 10)
    assert len(got) == 1
    assert len(got[0]) == MAX_SNIPPET_CHARS
