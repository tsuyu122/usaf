import math
import json
import time
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .config import USAFConfig
from .data import CppDataset, create_dataloader
from .importance import ImportanceScorer
from .selector import DynamicSelector
from .sparse_optim import SparseAdam
from .cache import ActivationCache
from .evaluate import Evaluator
from .utils import get_dml_device, count_parameters, estimate_optimizer_memory


class USAFFineTuner:
    def __init__(self, config: USAFConfig):
        self.config = config
        self.device = get_dml_device()
        self.cpu = torch.device("cpu")

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.writer = SummaryWriter(log_dir=config.log_dir)
        self._setup_model()
        self._setup_components()

    def _setup_model(self):
        print(f"Loading model {self.config.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        target_dtype = torch.float16

        load_kwargs = {
            "torch_dtype": target_dtype,
            "trust_remote_code": True,
        }

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **load_kwargs,
            )
        except Exception as e:
            print(f"Standard load failed ({e}), trying low_cpu_mem_usage...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                low_cpu_mem_usage=True,
                **load_kwargs,
            )

        if self.config.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except (AttributeError, NotImplementedError):
                print("Warning: gradient_checkpointing not supported")

        self.model = self.model.to(self.device)
        self.model.train()

        total = count_parameters(self.model)
        print(f"Model loaded: {total:,} total parameters ({total/1e9:.2f}B)")

    def _setup_components(self):
        self.scorer = ImportanceScorer(
            self.model,
            self.device,
            torch.float16,
            self.config.context_length,
        )
        self.selector = DynamicSelector(
            self.config.initial_active_k,
            self.config.reselect_every_n_steps,
            "topk",
        )
        self.cache = ActivationCache(device=self.cpu)
        self.evaluator = Evaluator(self.model, self.device, self.tokenizer)

        self.scores: dict[str, torch.Tensor] = {}
        self._optimizer: Optional[SparseAdam] = None

    def _get_all_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def _get_named_params(self) -> dict[str, torch.nn.Parameter]:
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def _load_or_compute_scores(self, dataloader) -> dict[str, torch.Tensor]:
        path = Path(self.config.importance_scores_path)
        if path.exists():
            print(f"Loading importance scores from {path}")
            return self.scorer.load_scores(str(path))

        print("Computing importance scores...")
        scores = self.scorer.compute_scores(
            dataloader,
            max_batches=0,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.scorer.save_scores(scores, str(path))
        return scores

    def _print_active_summary(self):
        if self._optimizer is None:
            return
        n = self._optimizer.num_active_params
        total = count_parameters(self.model)
        pct = (n / total) * 100
        mem = self._optimizer.optimizer_memory_mb
        print(f"  Active: {n:,} params ({pct:.4f}%) | Optimizer memory: {mem:.1f} MB")

    def train(
        self,
        train_dataloader,
        val_dataloader,
        resume_checkpoint: Optional[str] = None,
    ):
        config = self.config

        self.scores = self._load_or_compute_scores(train_dataloader)
        mask = self.selector.update_mask(self.scores, config.initial_active_k)

        self._optimizer = SparseAdam(
            self._get_named_params(),
            mask,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        self._print_active_summary()

        if config.use_activation_cache:
            self.cache.register_hooks(self.model)

        global_step = 0
        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()
        tokens_processed = 0

        for epoch in range(config.max_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            self.model.train()

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
            for batch_idx, batch in enumerate(pbar):
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
                if loss is None:
                    del input_ids, labels, outputs
                    continue

                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps

                loss.backward()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    if config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), config.grad_clip_norm
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item()
                epoch_batches += 1
                tokens_processed += input_ids.numel()

                del input_ids, labels, outputs

                if self.selector.should_reselect() and global_step > 0:
                    try:
                        new_scores = self.scorer.compute_scores(
                            train_dataloader, max_batches=10
                        )
                        new_mask = self.selector.update_mask(new_scores)
                        self._optimizer.reselect(self._get_named_params(), new_mask)
                    except Exception as e:
                        print(f"Warning: reselection failed: {e}")

                if global_step > 0 and global_step % config.log_every_n_steps == 0:
                    avg_loss = epoch_loss / max(epoch_batches, 1)
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_processed / max(elapsed, 1)
                    ppl = math.exp(avg_loss)

                    self.writer.add_scalar("train/loss", avg_loss, global_step)
                    self.writer.add_scalar("train/perplexity", ppl, global_step)
                    self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, global_step)

                    if self._optimizer:
                        self.writer.add_scalar(
                            "train/active_params", self._optimizer.num_active_params, global_step
                        )

                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        ppl=f"{ppl:.1f}",
                        active=f"{self._optimizer.num_active_params if self._optimizer else 0:,}",
                    )

                if global_step > 0 and global_step % config.eval_every_n_steps == 0:
                    val_results = self.evaluator.evaluate_perplexity(val_dataloader, max_batches=50)
                    val_loss = val_results["loss"]
                    val_ppl = val_results["perplexity"]

                    self.writer.add_scalar("val/loss", val_loss, global_step)
                    self.writer.add_scalar("val/perplexity", val_ppl, global_step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self._save_checkpoint(global_step, epoch, val_loss)
                    else:
                        patience_counter += 1

                    if patience_counter >= config.early_stopping_patience:
                        print(f"Early stopping at step {global_step}")
                        break

                if global_step > 0 and global_step % config.save_every_n_steps == 0:
                    self._save_checkpoint(global_step, epoch, epoch_loss / max(epoch_batches, 1))

            if patience_counter >= config.early_stopping_patience:
                break

            if config.use_activation_cache:
                self.cache.advance_step()

        if config.use_activation_cache:
            self.cache.remove_hooks()

        self.writer.close()
        elapsed = time.time() - start_time
        print(f"\nTraining complete: {elapsed:.0f}s, {tokens_processed} tokens, "
              f"{tokens_processed/max(elapsed,1):.0f} tokens/s")

    def _save_checkpoint(self, step: int, epoch: int, val_loss: float):
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "config": self.config,
        }
        if self._optimizer:
            ckpt["active_param_names"] = list(self._optimizer._active_ids)
        torch.save(ckpt, path / f"checkpoint-{step}.pt")
        try:
            self.model.save_pretrained(path / f"checkpoint-{step}")
        except Exception:
            pass
        print(f"  Checkpoint saved: step={step}")
