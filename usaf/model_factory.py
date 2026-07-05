"""Auto-detect MoE architecture from any HuggingFace model config."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import math

@dataclass
class MoEConfig:
    """Extracted MoE architecture parameters."""
    model_path: str = ""
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 2
    expert_intermediate: int = 0
    is_moe: bool = False
    
    # Parameter naming patterns
    expert_prefix: str = ""
    expert_param_names: List[str] = field(default_factory=lambda: ["gate_up_proj", "down_proj"])
    router_path: str = ""
    
    # Auto-configured
    train_from: int = 0
    max_trainable_layers: int = 0
    estimated_vram_gb: float = 0
    estimated_per_layer_gb: float = 0
    estimated_system_ram_gb: float = 0


def detect_model(model_path: str, vram_gb: float = 0, system_ram_gb: float = 0) -> MoEConfig:
    """Detect MoE architecture from a HuggingFace model path or local directory.
    
    Args:
        model_path: HuggingFace model ID or local path
        vram_gb: Available GPU VRAM in GB (0 = auto-detect)
        system_ram_gb: Available system RAM in GB (0 = auto-detect)
    
    Returns:
        MoEConfig with all detected parameters and auto-configured training settings.
    """
    from transformers import AutoConfig
    
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    config = MoEConfig(
        model_path=model_path,
        num_layers=getattr(cfg, 'num_hidden_layers', 0),
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads),
        head_dim=getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads),
        vocab_size=cfg.vocab_size,
    )
    
    # Detect MoE parameters
    config.num_experts = _detect_num_experts(cfg)
    config.num_experts_per_tok = _detect_experts_per_tok(cfg)
    config.expert_intermediate = _detect_expert_intermediate(cfg)
    config.is_moe = config.num_experts > 0
    
    if not config.is_moe:
        return config
    
    # Map parameter naming conventions
    config.expert_prefix, config.expert_param_names, rpath = _detect_param_names(cfg, model_path)
    config.router_path = rpath
    
    # Auto-configure training
    _auto_configure_training(config, vram_gb, system_ram_gb)
    
    return config


def _detect_num_experts(cfg) -> int:
    """Extract number of experts from config, handling different naming conventions."""
    for attr in ['num_experts', 'num_local_experts', 'n_routed_experts', 'moe_num_experts']:
        val = getattr(cfg, attr, None)
        if val is not None and val > 0:
            return val
    return 0


def _detect_experts_per_tok(cfg) -> int:
    """Extract number of active experts per token."""
    for attr in ['num_experts_per_tok', 'top_k', 'num_selected_experts', 'moe_top_k']:
        val = getattr(cfg, attr, None)
        if val is not None and val > 0:
            return val
    return 2  # default for most MoE models


def _detect_expert_intermediate(cfg) -> int:
    """Extract expert intermediate size."""
    for attr in ['moe_intermediate_size', 'expert_intermediate_size', 'intermediate_size']:
        val = getattr(cfg, attr, None)
        if val is not None and val > 0:
            return val
    return cfg.intermediate_size if hasattr(cfg, 'intermediate_size') else 0


def _detect_param_names(cfg, model_path) -> Tuple[str, List[str], str]:
    """Detect parameter naming conventions based on model architecture."""
    model_type = getattr(cfg, 'model_type', '').lower()
    architectures = getattr(cfg, 'architectures', [])
    arch_str = ' '.join(architectures).lower() if architectures else model_type
    
    # Qwen3-MoE / Qwen2-MoE
    if 'qwen' in arch_str or 'qwen' in model_type:
        return ("model.layers.{i}.mlp.experts", 
                ["gate_up_proj", "down_proj"],
                ".mlp.gate.weight")
    
    # Mixtral
    if 'mixtral' in arch_str:
        return ("model.layers.{i}.block_sparse_moe.experts",
                ["w1", "w2", "w3"],
                ".block_sparse_moe.gate.weight")
    
    # OLMoE
    if 'olmoe' in arch_str:
        return ("model.layers.{i}.mlp.experts",
                ["gate_proj", "up_proj", "down_proj"],
                ".mlp.gate.weight")
    
    # DeepSeek-MoE
    if 'deepseek' in arch_str:
        return ("model.layers.{i}.mlp.experts",
                ["gate_proj", "up_proj", "down_proj"],
                ".mlp.gate.weight")
    
    # Default: assume Qwen3-MoE style (most common)
    return ("model.layers.{i}.mlp.experts",
            ["gate_up_proj", "down_proj"],
            ".mlp.gate.weight")


def _auto_configure_training(config: MoEConfig, vram_gb: float, system_ram_gb: float):
    """Auto-configure which layers to train based on available memory."""
    import psutil
    import torch
    
    if vram_gb <= 0 and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if system_ram_gb <= 0:
        system_ram_gb = psutil.virtual_memory().total / 1e9
    
    config.estimated_vram_gb = vram_gb
    config.estimated_system_ram_gb = system_ram_gb
    
    # Calculate memory per trainable layer
    # Each expert param: hidden_size × expert_intermediate × 2 (gate+up) + expert_intermediate × hidden_size (down)
    # For fused gate_up: hidden_size × (expert_intermediate × 2)
    # For separate gate/up/down: use config to determine
    n_params = len(config.expert_param_names)
    if n_params >= 2:
        expert_bytes = (config.hidden_size * config.expert_intermediate * n_params * 
                       config.num_experts * 2) / 1e9  # fp16
    else:
        expert_bytes = 0.5  # fallback estimate
    
    # Resident mode keeps trainable experts in RAM (fp16) + q4 buffers + optimizer state
    resident_gb = expert_bytes * 0.5       # fp16 copy of trainable experts (~half kept resident)
    q4_gb = expert_bytes * 0.25            # q4 packed format
    optimizer_gb = expert_bytes * 0.5 * 2  # SparseAdam m+v
    overhead_gb = 0.5
    
    config.estimated_per_layer_gb = resident_gb + q4_gb + optimizer_gb + overhead_gb
    
    # Configure trainable layers
    usable_ram = system_ram_gb * 0.6  # leave 40% for OS + model
    config.max_trainable_layers = max(1, min(
        config.num_layers,
        int(usable_ram / max(config.estimated_per_layer_gb, 0.1))
    ))
    
    # Default: train top layers (closest to output, most task-specific)
    config.train_from = max(0, config.num_layers - config.max_trainable_layers)


def get_trainable_layers(config: MoEConfig, custom_train_from: Optional[int] = None) -> Set[int]:
    """Get the set of trainable layer indices."""
    start = custom_train_from if custom_train_from is not None else config.train_from
    return set(range(start, config.num_layers))


def get_param_patterns(config: MoEConfig) -> Dict[str, List[str]]:
    """Get parameter name patterns for the model.
    
    Returns:
        Dict mapping layer_index -> list of full parameter names for sparse training.
    """
    patterns = {}
    for li in range(config.num_layers):
        prefix = config.expert_prefix.format(i=li)
        names = [f"{prefix}.{pn}" for pn in config.expert_param_names]
        patterns[li] = names
    return patterns


def get_router_path(config: MoEConfig, layer_idx: int) -> str:
    """Get the router (gate) parameter path for a given layer."""
    return f"model.layers.{layer_idx}{config.router_path}"
