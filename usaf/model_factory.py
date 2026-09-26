"""Auto-detect MoE architecture from any HuggingFace model config."""
import os
from dataclasses import dataclass, field


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
    tie_word_embeddings: bool = False
    estimated_fixed_gb: float = 0.0
    num_experts: int = 0
    num_experts_per_tok: int = 2
    expert_intermediate: int = 0
    is_moe: bool = False

    expert_prefix: str = ""
    expert_param_names: list[str] = field(default_factory=lambda: ["gate_up_proj", "down_proj"])
    router_path: str = ""

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

    # A path that does not exist used to surface as whatever HuggingFace raised
    # for the string it was handed: "OSError: Repo id must use alphanumeric
    # chars, '-', '_' or '.'" - which says nothing about the missing
    # directory, and never mentions the path the user typed.
    looks_local = os.path.sep in model_path or (os.path.altsep or "") in model_path
    if looks_local and not os.path.isdir(model_path):
        raise SystemExit(f"model directory not found: {model_path}")

    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        cfg_path = os.path.join(model_path, "config.json")
        if looks_local and not os.path.exists(cfg_path):
            raise SystemExit(
                f"{model_path} has no config.json; that is not a model directory "
                f"(original error: {type(e).__name__}: {e})"
            ) from e
        # The file is there but AutoConfig still cannot read it: a config with no
        # model_type, a truncated download, a directory that is not a model. The raw
        # ValueError from deep inside configuration_auto names none of those.
        if looks_local:
            raise SystemExit(
                f"{cfg_path} is not a config transformers can read "
                f"({type(e).__name__}: {e}). A model directory needs a "
                f"config.json with a model_type key."
            ) from e
        raise

    config = MoEConfig(
        model_path=model_path,
        num_layers=getattr(cfg, 'num_hidden_layers', 0),
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads),
        head_dim=getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads),
        vocab_size=cfg.vocab_size,
    tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
    )

    config.num_experts = _detect_num_experts(cfg)
    config.num_experts_per_tok = _detect_experts_per_tok(cfg)
    config.expert_intermediate = _detect_expert_intermediate(cfg)
    config.is_moe = config.num_experts > 0

    if not config.is_moe:
        return config

    config.expert_prefix, config.expert_param_names, rpath = _detect_param_names(cfg, model_path)
    config.router_path = rpath

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
    return 2


def _detect_expert_intermediate(cfg) -> int:
    """Extract expert intermediate size."""
    for attr in ['moe_intermediate_size', 'expert_intermediate_size', 'intermediate_size']:
        val = getattr(cfg, attr, None)
        if val is not None and val > 0:
            return val
    return cfg.intermediate_size if hasattr(cfg, 'intermediate_size') else 0


def _detect_param_names(cfg, model_path) -> tuple[str, list[str], str]:
    """Detect parameter naming conventions based on model architecture.

    From transformers >= 4.53 every supported MoE family (Qwen3-MoE, Mixtral,
    OLMoE) exposes a *fused* expert container with 3D parameters named
    gate_up_proj / down_proj under ".mlp.experts". Older transformers used
    block_sparse_moe with per-expert w1/w2/w3 for Mixtral and separate
    gate_proj/up_proj/down_proj for OLMoE; that layout is no longer produced
    by the installed transformers, so the legacy branches are gone and every
    family resolves to the fused names below.
    """
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

    n_params = len(config.expert_param_names)
    if n_params >= 2:
        expert_bytes = (config.hidden_size * config.expert_intermediate * n_params *
                       config.num_experts * 2) / 1e9
    else:
        expert_bytes = 0.5

    resident_gb = expert_bytes * 0.5
    q4_gb = expert_bytes * 0.25
    optimizer_gb = expert_bytes * 0.5 * 2
    overhead_gb = 0.5

    # layers train, and the budget never counted them. ZAYA1-8B ties its head to
    # the input embedding, so that is one tensor of vocab x hidden - 262272 x
    # 2048 in fp16 is 1.07 GB, a fifteenth of a 15.6 GB card. The run that
    # estimated 14 trainable layers then died 65 minutes in trying to allocate
    # 256 MB with 194 MB free, because the layers it had budgeted for and the
    # embedding it had ignored together exceeded the card.
    _vocab = getattr(config, "vocab_size", 0) or 0
    _heads = 1 if getattr(config, "tie_word_embeddings", False) else 2
    fixed_gb = (_vocab * config.hidden_size * 2 * _heads) / 1e9 if _vocab else 0.0
    config.estimated_fixed_gb = fixed_gb

    # The attention weights of each trainable layer are also resident and are
    # not part of the expert tensors the estimate is built from: four projections
    # of hidden x hidden in fp16.
    attn_gb = (4 * config.hidden_size * config.hidden_size * 2) / 1e9
    config.estimated_per_layer_gb = (
        resident_gb + q4_gb + optimizer_gb + overhead_gb + attn_gb
    )

    # The embedding and the output head are on the device no matter how many
    usable_ram = system_ram_gb * 0.6
    max_by_ram = int(usable_ram / max(config.estimated_per_layer_gb, 0.1))

    # The safety factor is 0.8 rather than 0.9 because the tail of a real run is
    # fragmentation and the frozen activation cache, neither of which is a tensor
    # the estimate knows about. The 0.9 fitted the layer count on paper and
    # overflowed in practice.
    if vram_gb > 0:
        max_by_vram = int(((vram_gb * 0.8) - fixed_gb)
                         / max(config.estimated_per_layer_gb, 0.1))
        max_by_ram = min(max_by_ram, max(1, max_by_vram))

    config.max_trainable_layers = max(1, min(config.num_layers, max_by_ram))

    config.train_from = max(0, config.num_layers - config.max_trainable_layers)


def get_trainable_layers(config: MoEConfig, custom_train_from: int | None = None) -> set[int]:
    """Get the set of trainable layer indices.

    ``custom_train_from`` of 0 or None means "let the memory budget decide", which
    is what --train-from documents (its default is 0 and its help says 0=auto).
    That budget is what _auto_configure_training computed into max_trainable_layers
    and train_from, and it takes the last N layers so the deepest ones get the
    sparse treatment.

    Taking 0 as a literal start instead - which is what this did, because main()
    always passes the parsed int rather than None - overrode the budget with
    "from the first layer" and trained every layer of every model. The sizing
    work in _auto_configure_training was therefore inert on every default
    invocation, and a run that did not fit in VRAM found out at the first
    allocation instead of at the point where the budget was chosen.

    An empty result used to reach the caller and blow up as "min() arg is an
    empty sequence" while formatting a progress line. A train-from past the end
    of the model trains nothing, which is always a mistake, so say so here.
    """
    if custom_train_from is None or custom_train_from == 0:
        cap = config.max_trainable_layers
        if cap and cap > 0:
            start = max(0, config.num_layers - min(cap, config.num_layers))
        else:
            # No budget was computed, so honour whatever train_from says.
            start = config.train_from
    else:
        start = custom_train_from
    if start < 0:
        raise SystemExit(f"train-from must not be negative, got {start}")
    if start >= config.num_layers:
        raise SystemExit(
            f"train-from={start} is past the last layer: the model has "
            f"{config.num_layers} layers (0..{config.num_layers - 1}), so nothing "
            f"would be trained. Lower it to at most "
            f"{config.num_layers - 1}."
        )
    return set(range(start, config.num_layers))


def get_param_patterns(config: MoEConfig) -> dict[str, list[str]]:
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
