import json
from pathlib import Path
from typing import Dict, List, Optional

SYNTHETIC_TEXTS: List[str] = [
    "int main() { int x = 0; return x; }",
    "void foo(int a, int b) { return a + b; }",
    "template<typename T> T max(T a, T b) { return a > b ? a : b; }",
    "struct Node { int val; Node* next; };",
    "for (int i = 0; i < n; i++) { arr[i] = i * 2; }",
    "if (ptr != nullptr) { delete ptr; ptr = nullptr; }",
    "namespace math { double sqrt(double x) { return x * 0.5; } }",
    "auto lambda = [](int x) -> int { return x * x; };",
    "std::vector<int> v; v.push_back(42);",
    "enum Color { RED, GREEN, BLUE };",
    "switch (x) { case 0: return 'a'; default: return 'z'; }",
    "while (!q.empty()) { auto i = q.front(); q.pop(); }",
    "uint32_t hash(const char* s) { uint32_t h=0; while(*s)h=h*31+*s++; return h; }",
    "constexpr int SIZE = 256;",
    "class Vector { float x; float y; };",
    "int binary_search(int arr[], int l, int r, int x) { while (l <= r) { int m = l + (r - l) / 2; if (arr[m] == x) return m; if (arr[m] < x) l = m + 1; else r = m - 1; } return -1; }",
]

CPP_EXTS = (".cpp", ".h", ".hpp", ".c", ".cc", ".cxx", ".hxx")
MAX_SNIPPET_CHARS = 4096


def get_eval_texts(
    dataset: str = "synthetic-cpp",
    path: Optional[str] = None,
    max_samples: int = 64,
) -> List[str]:
    if dataset == "synthetic-cpp":
        return SYNTHETIC_TEXTS[:max_samples]
    if dataset in ("jsonl", "text") and path and Path(path).exists():
        texts: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if len(texts) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                if dataset == "jsonl":
                    try:
                        data = json.loads(line)
                        text = data.get("text") or data.get(
                            "content") or data.get("input") or line
                    except json.JSONDecodeError:
                        text = line
                else:
                    text = line
                if text:
                    texts.append(text[:MAX_SNIPPET_CHARS])
        return texts
    return SYNTHETIC_TEXTS[:max_samples]


class EvalDataset:
    def __init__(self, texts, tokenizer, seq_len=512, name="dataset"):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.name = name

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(text, truncation=True, max_length=self.seq_len, return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels, "text": text}


class SyntheticCppDataset(EvalDataset):
    def __init__(self, tokenizer, seq_len=512, max_samples=64):
        texts = SYNTHETIC_TEXTS[:max_samples]
        super().__init__(texts, tokenizer, seq_len, name="synthetic-cpp")
