from .cache import ActivationCache
from .checkpoint import export_merged_weights, get_checkpoint_metadata, load_sparse_checkpoint, save_sparse_checkpoint
from .config import USAFConfig
from .data import CppDataset, create_dataloader
from .evaluate import Evaluator
from .importance import ImportanceScorer
from .olmoe_dml import dml_experts_forward, dml_moe_block_forward, patch_olmoe_for_dml
from .olmoe_streaming import apply_captured_expert_grads, setup_streaming, sync_expert_grads_to_cpu
from .qwen3moe_dml import dml_qwen3_experts_forward, dml_qwen3_moe_block_forward, patch_qwen3moe_for_dml
from .selector import DynamicSelector, ThresholdSelector, TopKSelector
from .sparse_optim import SparseAdam

_HAS_QUANTIZATION: bool = False
_HAS_MOE_LOADER: bool = False

try:
    from .quantization import (
        dequantize_4bit,
        dequantize_state_dict,
        dequantize_with_outliers,
        estimate_quantized_size,
        estimate_quantized_state_dict_size,
        quantize_4bit,
        quantize_state_dict,
        quantize_with_outliers,
        reconstruction_error,
    )
    _HAS_QUANTIZATION = True
except ImportError:
    pass

try:
    from .moe_loader import (
        QuantizedExpertCache,
        apply_captured_expert_grads,
        get_quantized_cache,
        load_and_stream,
        load_quantized_state_dict,
        save_quantized_state_dict,
        setup_quantized_streaming,
    )
    _HAS_MOE_LOADER = True
except ImportError:
    pass

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

from .eval import (
    BenchmarkConfig,
    BenchmarkResults,
    compare_reports,
    compute_perplexity,
    run_benchmark,
    save_report,
)

# These are deliberate re-exports of the public API. Registering them in
# __all__ is what makes them intentional rather than dead imports - without
# it linters flag them and, worse, a reader assumes the package does not
# expose them at all.
__all__ += [
    "export_merged_weights",
    "get_checkpoint_metadata",
    "load_sparse_checkpoint",
    "save_sparse_checkpoint",
    "BenchmarkConfig",
    "BenchmarkResults",
    "compare_reports",
    "compute_perplexity",
    "run_benchmark",
    "save_report",
]
