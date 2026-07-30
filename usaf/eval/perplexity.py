import math
import time
from typing import Dict, List

import torch


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    seq_len: int = 512,
    batch_size: int = 1,
    desc: str = "Evaluating",
    verbose: bool = True,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_samples = len(texts)
    t0 = time.time()

    for i in range(0, total_samples, batch_size):
        batch_texts = texts[i : i + batch_size]

        for text in batch_texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            labels = input_ids.clone()

            try:
                outputs = model(input_ids=input_ids, labels=labels)
            except TypeError:
                outputs = model(input_ids=input_ids, labels=labels, pixel_values=None)

            loss = outputs.loss
            if loss is not None:
                batch_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * max(batch_tokens, 1)
                total_tokens += max(batch_tokens, 1)

        if verbose and total_samples > 1:
            n_processed = min(i + batch_size, total_samples)
            if n_processed % max(1, total_samples // 4) == 0:
                elapsed = time.time() - t0
                ppl_now = math.exp(total_loss / max(total_tokens, 1))
                print(f"  {desc}: {n_processed}/{total_samples} | current PPL {ppl_now:.2f}")

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    elapsed = time.time() - t0

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "tokens": total_tokens,
        "samples": total_samples,
        "time_s": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / max(elapsed, 0.001), 1),
    }
