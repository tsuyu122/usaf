import os
import random
from pathlib import Path
from typing import Iterator, Optional

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


def collect_source_files(root_dir: str, extensions: tuple, max_size_mb: int = 1) -> list[str]:
    max_bytes = max_size_mb * 1024 * 1024
    files = []
    root = Path(root_dir)
    for ext in extensions:
        for fpath in root.rglob(f"*{ext}"):
            if fpath.is_file():
                try:
                    size = fpath.stat().st_size
                except OSError:
                    continue
                if 0 < size <= max_bytes:
                    files.append(str(fpath))
    return files


def read_file_content(filepath: str, deduplicate_lines: bool = True) -> str:
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    content = None
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if content is None:
        return ""
    if deduplicate_lines:
        lines = content.split("\n")
        seen = set()
        unique = []
        for line in lines:
            stripped = line.rstrip()
            if stripped and stripped not in seen:
                seen.add(stripped)
            unique.append(line)
        content = "\n".join(unique)
    return content


def tokenize_text(
    text: str,
    tokenizer: AutoTokenizer,
    context_length: int = 2048,
    overlap: int = 128,
):
    stride = context_length - overlap
    encoding = tokenizer(
        text,
        truncation=False,
        return_tensors=None,
        add_special_tokens=True,
    )
    input_ids = encoding["input_ids"]
    chunks_yielded = 0
    for start in range(0, len(input_ids) - context_length + 1, stride):
        chunk = input_ids[start : start + context_length]
        chunks_yielded += 1
        yield {"input_ids": chunk, "labels": chunk}
    if len(input_ids) > 0 and chunks_yielded == 0:
        chunk = input_ids[:context_length]
        pad_len = context_length - len(chunk)
        chunk = chunk + [tokenizer.pad_token_id or 0] * pad_len
        yield {"input_ids": chunk, "labels": chunk}


def preprocess_dataset(
    cloned_projects_dir: str,
    tokenizer: AutoTokenizer,
    context_length: int = 2048,
    chunk_overlap: int = 128,
    cpp_extensions: tuple = (".h", ".hpp", ".cpp", ".c", ".cc", ".cxx", ".hxx"),
    max_file_size_mb: int = 1,
    deduplicate_lines: bool = True,
    train_split: float = 0.95,
    shuffle_repos: bool = True,
    seed: int = 42,
) -> DatasetDict:
    import tempfile
    import shutil

    random.seed(seed)
    files = collect_source_files(cloned_projects_dir, cpp_extensions, max_file_size_mb)
    if shuffle_repos:
        repo_groups = {}
        for f in files:
            repo = str(Path(f).relative_to(cloned_projects_dir)).split(os.sep)[0]
            repo_groups.setdefault(repo, []).append(f)
        repo_names = list(repo_groups.keys())
        random.shuffle(repo_names)
        files = []
        for repo in repo_names:
            files.extend(sorted(repo_groups[repo]))

    temp_root = Path(os.environ.get("TMPDIR", cloned_projects_dir)) / ".." / ".."
    temp_root = temp_root.resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="usaf_preprocess_", dir=str(temp_root)))
    batch_size = 50000
    batch_chunks = []
    batch_id = 0

    try:
        pbar = tqdm(total=len(files), desc="Tokenizing")
        for fpath in files:
            content = read_file_content(fpath, deduplicate_lines)
            if not content.strip():
                pbar.update(1)
                continue
            for chunk_data in tokenize_text(content, tokenizer, context_length, chunk_overlap):
                batch_chunks.append(chunk_data)
                if len(batch_chunks) >= batch_size:
                    ds = Dataset.from_list(batch_chunks)
                    ds.to_parquet(str(temp_dir / f"batch_{batch_id:04d}.parquet"))
                    batch_chunks = []
                    batch_id += 1
            pbar.update(1)
        pbar.close()

        if batch_chunks:
            ds = Dataset.from_list(batch_chunks)
            ds.to_parquet(str(temp_dir / f"batch_{batch_id:04d}.parquet"))
            batch_id += 1
            batch_chunks = []

        parquet_files = sorted(temp_dir.glob("batch_*.parquet"))
        if not parquet_files:
            raise RuntimeError("No chunks were generated from source files")

        all_datasets = []
        for pf in tqdm(parquet_files, desc="Merging batches"):
            all_datasets.append(Dataset.from_parquet(str(pf)))

        dataset = concatenate_datasets(all_datasets)
        split = dataset.train_test_split(test_size=1.0 - train_split, seed=seed)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return DatasetDict({"train": split["train"], "validation": split["test"]})


class CppDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset: Dataset):
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        CppDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=True,
    )
