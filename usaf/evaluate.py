import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import get_dml_device


class Evaluator:
    def __init__(
        self,
        model,
        device: torch.device,
        tokenizer=None,
    ):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    @torch.no_grad()
    def evaluate_perplexity(self, dataloader: DataLoader, max_batches: int = 0) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating perplexity")):
            if max_batches > 0 and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)

            try:
                outputs = self.model(input_ids=input_ids, labels=labels)
            except TypeError:
                outputs = self.model(input_ids=input_ids, labels=labels, pixel_values=None)

            loss = outputs.loss
            if loss is not None:
                batch_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

            del input_ids, labels, outputs

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(avg_loss)

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
        }

    @torch.no_grad()
    def generate_sample(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
    ) -> str:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for generation")
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
