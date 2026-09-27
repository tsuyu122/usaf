"""Auto-detect MoE architecture from any HuggingFace model config."""
import glob
import json
import os
import re
import struct
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



def _keys_from_single_safetensors(model_path) -> list[str]:
    """Tensor names from a one-file checkpoint, without loading any weight.

    safetensors keeps its key table in the first 8 bytes as a JSON length, so
    this costs a header read rather than a 1.6 GB download. Reading the header
    is also the only way to fail early and cheaply on a model whose names this
    code does not understand, which is much better than finding out at the
    first optimizer step.
    """
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if len(files) != 1:
        # Several shards and no index is a layout this has never been asked
        # about; say so rather than guessing across the shards.
        return []
    try:
        with open(files[0], "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            if n > 100_000_000:
                return []
            header = json.loads(fh.read(n).decode("utf-8"))
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return []
    return [k for k in header if k != "__metadata__"]


def _loaded_router_name(on_disk: str) -> str:
    """The router name on the loaded module, given its name in the file.

    GraniteMoe stores ``router.layer.weight`` and the converter flattens it to
    ``router.weight`` when the module is built. The detector has to answer for
    the module, because the module is what the trainer resolves names in - and
    a path that exists only on disk is a path that resolves to nothing at
    runtime, with no error, which is the ZAYA failure in different clothes.

    Only a path with that middle ``layer`` is collapsed. Anything else is passed
    through untouched, because rewriting a name this code does not understand
    is worse than reporting the name it was given.
    """
    parts = on_disk.split(".")
    if (len(parts) >= 3 and parts[-3] == "router" and parts[-2] == "layer"
            and parts[-1] == "weight"):
        return ".".join(parts[:-3] + [parts[-3], parts[-1]])
    return on_disk


def _router_disk_name(loaded: str) -> str:
    """The router name in the file, given the name on the loaded module.

    The inverse of _loaded_router_name, and needed for the same reason from the
    other side. The trainer walks the keys of the checkpoint and fills the model
    from them, so a module name with no key of its own in the file is simply not
    loaded: the parameter keeps the meta device it was built with and the first
    forward dies naming a weight that was sitting in the file all along. For
    GraniteMoe that is 24 routers, one per layer.

    Only router.weight gains the layer component. Everything else is returned as
    given, for the same reason as above: rewriting a name this code does not
    understand is worse than reporting it.
    """
    parts = loaded.split(".")
    if len(parts) >= 2 and parts[-2] == "router" and parts[-1] == "weight":
        return ".".join(parts[:-2] + ["router", "layer", "weight"])
    return loaded


def _expert_names_from_index(model_path) -> tuple[str, list[str], str] | None:
    """Read the expert and router names off a checkpoint, or None.

    Every answer here is checked against the real file rather than assumed from
    the family name. Returns None when there is no index, no single file to read,
    or no expert tensor at all - the caller then falls back and says so.
    """
    if not os.path.isdir(model_path):
        return None

    keys: list[str] = []
    for index_path in sorted(glob.glob(os.path.join(model_path, "*.index.json"))):
        try:
            with open(index_path, encoding="utf-8") as fh:
                keys = list(json.load(fh).get("weight_map", {}).keys())
        except (OSError, ValueError):
            return None
        break
    if not keys:
        # A single-file checkpoint has no index. The names are still readable,
        # and GraniteMoe is exactly this shape: one 1.6 GB safetensors, so the
        # index path alone would have declared it undetectable.
        keys = _keys_from_single_safetensors(model_path)
    if not keys:
        return None

    # The names on disk and the names after loading are not always the same.
    # GraniteMoe stores its experts as input_linear / output_linear and the
    # converter renames them to gate_up_proj / down_proj on the way in, so a
    # model can be perfectly loadable and still match neither spelling here.
    fused = re.compile(
        r"^model\.layers\.\d+\.(.+)\.experts\.(gate_up_proj|down_proj)$"
    )
    stored = re.compile(
        r"^model\.layers\.\d+\.(.+)\.(input_linear|output_linear)\.weight$"
    )
    rename = {"input_linear": "gate_up_proj", "output_linear": "down_proj"}

    prefixes: list[str] = []
    names: list[str] = []
    for k in keys:
        m = fused.match(k)
        if m:
            prefixes.append(m.group(1))
            names.append(m.group(2))
            continue
        m = stored.match(k)
        if m:
            prefixes.append(m.group(1))
            names.append(rename[m.group(2)])
    if not prefixes or not names:
        return None

    # Every layer must agree, or the prefix is not the prefix. One layer under
    # mlp and another under block_sparse_moe is not a naming convention, it is
    # a model this does not understand, and picking one trains half of it.
    if len(set(prefixes)) != 1:
        return None

    ordered = [n for n in ("gate_up_proj", "down_proj") if n in names]
    if len(ordered) < 2:
        return None
    block = prefixes[0]

    # The router is the sibling that is not an expert tensor. input_linear has
    # no "expert" in its name, so a name-only filter returns it and the router
    # path ends up pointing at a weight matrix.
    sibling = re.compile(
        r"^model\.layers\.\d+\." + re.escape(block) + r"\.(.+)$"
    )
    expert_tensors = {"experts." + n for n in ordered}
    expert_tensors.update({"input_linear.weight", "output_linear.weight"})
    router = None
    for k in keys:
        m = sibling.match(k)
        if not m:
            continue
        rel = m.group(1)
        if rel in expert_tensors or "expert" in rel:
            continue
        router = rel
        break
    if router is None:
        return None

    prefix = "model.layers.{i}." + block + ".experts"
    return prefix, ordered, "." + block + "." + _loaded_router_name(router)


def _detect_param_names(cfg, model_path) -> tuple[str, list[str], str]:
    """Detect parameter naming conventions from the checkpoint itself.

    Reading the names out of the weight file is the only way this is right for a
    model nobody has seen before, and the two families that motivated the old
    table are the argument for it: ZAYA1-8B puts its experts under mlp.experts
    with a router at mlp.gate.weight, GraniteMoe puts the same tensors under
    block_sparse_moe.experts with a router at block_sparse_moe.router.weight, and
    both are MoE while neither matches the other.

    A model that trains is a stronger requirement than a model that is on the
    list, so the file is consulted first and the historical names are the
    fallback - used only when there is no file to read, and reported as such.
    """
    found = _expert_names_from_index(model_path)
    if found is not None:
        return found
    # No checkpoint to read. Returning the Qwen layout here is a guess that has
    # no evidence behind it, and a wrong guess produces a model that loads and
    # trains nothing - the failure this function exists to prevent - so the
    # caller is told the names were not verified rather than handed a lie.
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


def _candidates(loaded: str) -> list[str]:
    """The file spellings this module name could have, best guess first.

    Two things vary independently between releases: whether there is an .experts
    container at all, and whether the tensor inside it is stored under its loaded
    name or the converted one. The order is the answer on this machine first, and
    every other combination after, so a release that changed one thing still
    resolves.
    """
    parts = loaded.split(".")
    tail = parts[-1]
    stems = []
    if len(parts) >= 2 and parts[-2] == "experts":
        head = parts[:-2]
        stems.append(head)
        stems.append(parts[:-1])
    else:
        stems.append(parts[:-1])
        stems.append(parts[:-2] + ["experts"] if len(parts) >= 2 else None)
    out = []
    for stem in stems:
        if stem is None:
            continue
        for name in (tail, _STORED_TAIL.get(tail)):
            if name is None:
                continue
            out.append(".".join(stem + [name]))
            if stem and stem[-1] == "router" and name == "weight":
                out.append(".".join(stem + ["router", "layer", "weight"]))
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def resolve_stored_name(loaded: str, keys) -> str:
    """The key this module name has in a checkpoint that holds `keys`.

    Asking the file is the only way to know: a checkpoint written by one release
    and loaded by another does not rename the same way, and the export has to use
    the spelling that is actually in the file or from_pretrained fills the
    difference with fresh random weights and reports success.
    """
    for cand in _candidates(loaded):
        if cand in keys:
            return cand
    return _stored_name(loaded)


# The tail a tensor is stored under, and the one it is loaded under. GraniteMoe
# renames on the way in, so the file and the module disagree about every expert
# weight and about the router, and code that writes the module spelling produces
# a directory that loads into a model full of fresh random experts.
_STORED_TAIL = {
    "gate_up_proj": "input_linear.weight",
    "down_proj": "output_linear.weight",
}

# The reverse, for a module whose release kept the converted names. Which of the
# two a release uses is not something the checkpoint can tell you: the file says
# the same thing either way.
_LOADED_TAIL = {
    "input_linear.weight": "gate_up_proj",
    "output_linear.weight": "down_proj",
    "input_linear": "gate_up_proj",
    "output_linear": "down_proj",
}


def _stored_name(loaded: str) -> str:
    """The name this tensor has in the file, given the name on the module.

    The inverse of _loaded_router_name for the router, and the same treatment for
    the expert weights, which the quantiser had discovered separately and kept to
    itself. Both questions have one answer - what is the tensor called in the file
    - and two copies of the answer is how they drift apart.
    """
    parts = loaded.split(".")
    if parts and parts[-1] in _STORED_TAIL:
        return ".".join(parts[:-1] + [_STORED_TAIL[parts[-1]]])
    return _router_disk_name(loaded)


def _name_candidates(prefix: str, pn: str) -> list[str]:
    """How a tensor called pn under this prefix could be spelled on a module.

    Four shapes, because two things vary independently: whether the converted name
    or the stored one survived, and whether the tensor is a Parameter or sits in a
    submodule of its own - which is what makes named_parameters report
    input_linear.weight rather than input_linear. A release may present any of
    the four, and the list is a lookup against the model rather than a guess
    about which release is installed: it either hits or it does not.
    """
    stems = [pn]
    for spelling in (pn, _STORED_TAIL.get(pn), _LOADED_TAIL.get(pn)):
        if spelling and spelling not in stems:
            stems.append(spelling)
    out = []
    for stem in stems:
        for tail in (stem, stem + ".weight"):
            if prefix + "." + tail not in out:
                out.append(prefix + "." + tail)
    return out

def resolve_expert_layout(model, cfg: MoEConfig) -> MoEConfig:
    """Point the expert prefix and the expert names at what this model has.

    The checkpoint answers neither question. A checkpoint names a tensor by the
    key it was saved under; which module holds it and under what name are both
    properties of the transformers release, and they change independently - one
    release keeps an .experts container and the loaded names, another drops the
    container and keeps the stored names, and the file is identical in both.

    So the prefix and the names are looked up on the model. Asking the model is
    the only source that knows, and reading the file instead has been wrong four
    times: it produces a run with no expert modules, then a run whose experts it
    cannot find, then a run that leaves every router on the meta device, each time
    while the loss curve still looks like training.
    """
    params = set(dict(model.named_parameters()))
    base = cfg.expert_prefix

    prefixes = [base]
    if base.endswith(".experts"):
        prefixes.append(base[: -len(".experts")])
    else:
        prefixes.append(base + ".experts")

    def names_of(prefix, pns):
        out = set()
        for li in range(cfg.num_layers):
            for pn in pns:
                out.update(_name_candidates(prefix.format(i=li), pn))
        return out

    # A working prefix is one where at least one spelling of the first tensor of
    # every layer is present. All of them or none: a release does not rename half
    # its layers.
    for cand in prefixes:
        names = names_of(cand, [cfg.expert_param_names[0]])
        if not names or not (names & params):
            continue
        ordered = []
        head = cand.format(i=0)
        for pn in cfg.expert_param_names:
            # The name relative to the prefix. Taking only the last component
            # would turn input_linear.weight into weight, which names a tensor
            # that does not exist on a module that registers it inside a
            # submodule of its own.
            hit = next(
                (c[len(head) + 1:]
                 for c in _name_candidates(head, pn) if c in params),
                None,
            )
            if hit is None:
                ordered = []
                break
            ordered.append(hit)
        if not ordered:
            continue
        cfg.expert_prefix = cand
        cfg.expert_param_names = ordered
        return cfg

    sample = sorted(
        n for n in params if n.startswith("model.layers.0."))[:24]
    raise SystemExit(
        f"the expert layout {cfg.expert_prefix} {cfg.expert_param_names} "
        f"resolves to no parameters on this model.\n  tried: {prefixes}"
        f"\n  layer 0 parameters: {sample}"
    )
