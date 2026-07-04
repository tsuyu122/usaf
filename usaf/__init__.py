from .config import USAFConfig
from .data import CppDataset, create_dataloader
from .importance import ImportanceScorer
from .selector import TopKSelector, ThresholdSelector, DynamicSelector
from .sparse_optim import SparseAdam
from .cache import ActivationCache
from .trainer import USAFFineTuner
from .evaluate import Evaluator
from .olmoe_dml import patch_olmoe_for_dml, dml_experts_forward, dml_moe_block_forward
from .olmoe_streaming import setup_streaming, apply_captured_expert_grads, sync_expert_grads_to_cpu
from .qwen3moe_dml import patch_qwen3moe_for_dml, dml_qwen3_experts_forward, dml_qwen3_moe_block_forward

try:
    from .quantization import (
        quantize_4bit,
        dequantize_4bit,
        quantize_state_dict,
        dequantize_state_dict,
        estimate_quantized_size,
        estimate_quantized_state_dict_size,
        reconstruction_error,
        quantize_with_outliers,
        dequantize_with_outliers,
    )
    _HAS_QUANTIZATION = True
except ImportError:
    _HAS_QUANTIZATION = False

try:
    from .moe_loader import (
        QuantizedExpertCache,
        save_quantized_state_dict,
        load_quantized_state_dict,
        setup_quantized_streaming,
        get_quantized_cache,
        apply_captured_expert_grads,
        load_and_stream,
    )
    _HAS_MOE_LOADER = True
except ImportError:
    _HAS_MOE_LOADER = False

__all__ = [
    "USAFConfig",
    "CppDataset",
    "create_dataloader",
    "ImportanceScorer",
    "TopKSelector",
    "ThresholdSelector",
    "DynamicSelector",
    "SparseAdam",
    "ActivationCache",
    "USAFFineTuner",
    "Evaluator",
    "patch_olmoe_for_dml",
    "dml_experts_forward",
    "dml_moe_block_forward",
    "setup_streaming",
    "apply_captured_expert_grads",
    "sync_expert_grads_to_cpu",
    "patch_qwen3moe_for_dml",
    "dml_qwen3_experts_forward",
    "dml_qwen3_moe_block_forward",
]

if _HAS_QUANTIZATION:
    __all__ += [
        "quantize_4bit",
        "dequantize_4bit",
        "quantize_state_dict",
        "dequantize_state_dict",
        "estimate_quantized_size",
        "estimate_quantized_state_dict_size",
        "reconstruction_error",
        "quantize_with_outliers",
        "dequantize_with_outliers",
    ]

if _HAS_MOE_LOADER:
    __all__ += [
        "QuantizedExpertCache",
        "save_quantized_state_dict",
        "load_quantized_state_dict",
        "setup_quantized_streaming",
        "get_quantized_cache",
        "apply_captured_expert_grads",
        "load_and_stream",
    ]
