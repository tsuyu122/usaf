from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class USAFConfig:
    model_id: str = "local_model_qwen"
    tokenizer_id: str = "local_model_qwen"
    dtype: str = "fp16"
    context_length: int = 2048

    cloned_projects_dir: str = "cloned_projects"
    preprocessed_dataset_dir: str = "data/preprocessed"
    importance_scores_path: str = "data/importance_scores.pt"
    cache_dir: str = "data/activation_cache"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    cpp_extensions: tuple = (".h", ".hpp", ".cpp", ".c", ".cc", ".cxx", ".hxx")
    max_file_size_mb: int = 1
    deduplicate_lines: bool = True
    train_split: float = 0.95
    val_split: float = 0.05
    chunk_overlap: int = 128
    shuffle_repos: bool = True

    initial_active_k: int = 400_000
    active_percentile: Optional[float] = None
    reselect_every_n_steps: int = 500
    initial_selection_epochs: int = 1

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_epochs: int = 10
    grad_clip_norm: float = 1.0

    use_activation_cache: bool = True
    cache_on_device: bool = False

    gradient_checkpointing: bool = True

    early_stopping_patience: int = 3
    eval_every_n_steps: int = 100
    log_every_n_steps: int = 10
    save_every_n_steps: int = 500
    seed: int = 42

    def resolve_paths(self, base_dir: Optional[Path] = None) -> "USAFConfig":
        if base_dir is None:
            base_dir = Path.cwd()
        base = Path(base_dir)
        for attr in [
            "cloned_projects_dir",
            "preprocessed_dataset_dir",
            "importance_scores_path",
            "cache_dir",
            "checkpoint_dir",
            "log_dir",
        ]:
            p = base / getattr(self, attr)
            setattr(self, attr, str(p))
        return self
