import torch
import torch_directml


def get_dml_device(device_id: int = 0) -> torch.device:
    return torch_directml.device(device_id)


def load_model_to_dml(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move model to DML device.

    Do NOT use low_cpu_mem_usage=True when loading — it enables the accelerate
    dispatch which creates thousands of internal steps and fails with
    'RuntimeError: unknown error' on DML. Load without it, then call this.
    """
    model.to(device)
    return model


def get_cpu_device() -> torch.device:
    return torch.device("cpu")


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    if device.type == "privateuseone":
        return torch.float16
    return torch.float32


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_optimizer_memory(num_active_params: int, dtype: torch.dtype = torch.float32) -> int:
    bytes_per_param = 4 if dtype == torch.float32 else 2
    return num_active_params * bytes_per_param * 2


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
