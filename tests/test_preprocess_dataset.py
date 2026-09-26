"""The dataset that comes back has to still be readable.

preprocess_dataset writes the tokenized chunks to parquet in a temp directory
and removes that directory in a finally block. A datasets Dataset can hold a
memory-mapped arrow table, in which case the object it returns is a view onto
files that no longer exist - and the failure lands far from the code that
caused it, on the first read, possibly in a different process.

This is the main C++ training path: a corpus of source files, tokenized into
chunks, written out, merged, split, and handed back. Every step of it is easy
to get right and none of it is exercised by a unit test.
"""

import pytest

pytestmark = pytest.mark.skipif(
    pytest.importorskip("datasets", reason="datasets not installed") is None,
    reason="datasets not installed",
)


class _Tok:
    pad_token_id = 0

    def __call__(self, text, **kw):
        return {"input_ids": list(range(1, 60))}


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "repos"
    (root / "projA").mkdir(parents=True)
    (root / "projB").mkdir(parents=True)
    (root / "projA" / "a.cpp").write_text("int a;\n" * 20, encoding="utf-8")
    (root / "projB" / "b.cpp").write_text("int b;\n" * 20, encoding="utf-8")
    (root / "projA" / "notes.md").write_text("nao e fonte", encoding="utf-8")
    return str(root)


def test_the_dataset_that_comes_back_can_actually_be_read(corpus):
    from usaf.data import preprocess_dataset

    out = preprocess_dataset(corpus, _Tok(), context_length=16, chunk_overlap=4,
                             train_split=0.8, shuffle_repos=False)
    assert "train" in out and "validation" in out

    # Read every row of both splits. If the table was memory-mapped onto a
    # temp directory that no longer exists, this is where it fails.
    total = 0
    for split in ("train", "validation"):
        ds = out[split]
        for i in range(len(ds)):
            item = ds[i]
            assert len(item["input_ids"]) == 16, (split, i, len(item["input_ids"]))
            assert len(item["labels"]) == 16, (split, i)
            total += 1
    assert total > 0, "the split came back empty"


def test_the_two_splits_partition_the_chunks(corpus):
    from usaf.data import preprocess_dataset

    out = preprocess_dataset(corpus, _Tok(), context_length=16, chunk_overlap=4,
                             train_split=0.8, shuffle_repos=False)
    assert len(out["train"]) > 0
    assert len(out["validation"]) > 0
    assert len(out["train"]) > len(out["validation"])


def test_a_directory_with_no_source_files_says_so(tmp_path):
    """"No chunks were generated" blames the tokenizer, but the cause is that
    there was nothing to read. Those are different problems."""
    from usaf.data import preprocess_dataset

    empty = tmp_path / "vazio"
    empty.mkdir()
    (empty / "notes.md").write_text("nao e fonte", encoding="utf-8")
    with pytest.raises(RuntimeError) as ei:
        preprocess_dataset(str(empty), _Tok(), context_length=16, chunk_overlap=4,
                           train_split=0.8, shuffle_repos=False)
    assert "no chunks" in str(ei.value).lower()
