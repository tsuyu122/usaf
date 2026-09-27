"""Route GraniteMoe experts through the sparse USAF forward.

GraniteMoeExperts is not one of the containers USAF replaces, so its forward
ran the dense path: every expert for every token. Nothing raises, the loss is
computed, and the run reports progress while no expert gradient is ever
captured - the training that produced "Active: 0/0" on ZAYA1, reached here by
a different route.

The forward it needs is the one the Qwen3-MoE patch installs. The signature is
already the same, (hidden_states, top_k_index, top_k_weights), and so are the
weights it reads: gate_up_proj and down_proj, both stacked over the expert
dimension, and num_experts. Granite was renamed onto the fused layout rather
than onto a different one, which is why this is a class swap and not a
translation.

The class is found by shape rather than by name. It is a module attribute in
some transformers releases and absent in others, where the experts live under
a different module path - and pinning a name turns a version difference into a
run that dies in the first thirty seconds, before it has said what it found.
Nothing here raises: a release whose layout is one this does not recognise gets
a line in the log, and the importance pass two minutes later is the check that
actually matters - it is the one that reports Active: 0/0 in plain words.

The router is left alone, because USAF trains the router as an ordinary
parameter rather than through this path.
"""

import inspect

import torch

_PATCHED: dict[str, object] = {}
_PATCHED_VALUES: set[int] = set()

_CANDIDATE_MODULES = (
    "transformers.models.granitemoe.modeling_granitemoe",
    "transformers.models.granitemoe.modeling_granite_moe",
    "transformers.models.granitemoe",
)


def _looks_like_experts(cls) -> bool:
    """An expert container: a module that owns the stacked expert weights.

    Two checks, because either alone matches the wrong thing. The names in
    __init__ decide which release it is - one builds the container with the
    fused names, another with input_linear and output_linear - and the block
    that holds the container mentions both of either pair. The forward
    signature is what tells them apart, and it is the honest check: the
    container takes the hidden states plus the routing the router just
    produced, the block takes the hidden states alone.

    Matching on the names alone patched the block. The block's forward - the
    one that runs the router - was replaced with a three-argument expert
    forward, and the run died on a TypeError one line later, after printing
    that the experts had been routed sparse. Checking for nn.Parameter in
    __init__ is closer but still a guess about how the next release spells
    the construction; the signature is the class telling us what it does.
    """
    if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
        return False
    try:
        src = inspect.getsource(cls.__init__)
    except (OSError, TypeError, ValueError):
        return False

    pairs = (("gate_up_proj", "down_proj"),
             ("input_linear", "output_linear"),)
    if not any(a in src and b in src for a, b in pairs):
        return False

    # Can it actually be handed the routing? That is the question the sparse
    # forward asks of it, so it is the question worth asking the class, and it
    # survives a release that adds an optional argument - which counting
    # parameters does not, and which is how the check below started refusing the
    # container it was written for.
    #
    # self, the hidden states, and the routing the router just produced.
    try:
        sig = inspect.signature(cls.forward)
        named = [p for p in sig.parameters.values()
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        sig.bind(
            object(), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    except (TypeError, ValueError):
        return False
    # A *args forward binds anything, and nn.Module.forward is exactly that, so
    # binding alone is not enough - the names have to be there as well.
    return len(named) >= 3

def _candidate_modules():
    """Every module GraniteMoe could have put its expert container in.

    The three pinned paths come first because they are where it has been, and
    the package walk after them so that a release which moved the class still
    finds it. Guessing a path is how the previous version of this file turned a
    transformers version difference into a crash thirty seconds into every run.
    """
    for name in _CANDIDATE_MODULES:
        try:
            yield __import__(name, fromlist=["*"])
        except ImportError:
            continue
    try:
        import pkgutil

        import transformers.models.granitemoe as pkg
    except ImportError:
        return
    for info in pkgutil.iter_modules(pkg.__path__):
        try:
            yield __import__(f"{pkg.__name__}.{info.name}", fromlist=["*"])
        except ImportError:
            continue


def cls_already_patched(cls) -> bool:
    """This exact class was patched already, under whatever name it was found by."""
    return cls in _PATCHED_VALUES


def _find_expert_classes():
    """Every expert container GraniteMoe has, found by what it holds.

    More than one is possible across releases - some keep a separate class, some
    inline the experts into the block - and patching only the first would leave
    a second container running dense without saying so.
    """
    found = []
    seen = set()
    for mod in _candidate_modules():
        for attr, obj in vars(mod).items():
            if not _looks_like_experts(obj):
                continue
            if not obj.__module__.startswith("transformers.models.granitemoe"):
                continue
            if id(obj) in seen or id(obj) in _PATCHED_VALUES:
                continue
            seen.add(id(obj))
            found.append((attr, obj))
    return found


def patch_granite_for_dml() -> list[str]:
    """Install the sparse expert forward. Returns the class names it patched."""
    from usaf.qwen3moe_dml import dml_qwen3_experts_forward

    patched = []
    for attr, cls in _find_expert_classes():
        if getattr(cls.forward, "_usaf_sparse", False):
            patched.append(attr)
            continue
        original = cls.forward

        def sparse_forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            return dml_qwen3_experts_forward(
                self, hidden_states, top_k_index, top_k_weights)

        sparse_forward._usaf_sparse = True
        _PATCHED[attr] = original
        _PATCHED_VALUES.add(id(cls))
        cls.forward = sparse_forward
        patched.append(attr)

    if patched:
        print(f"  granite experts routed sparse: {patched}", flush=True)
    else:
        print("  granite experts: no container found, nothing patched", flush=True)
    return patched


def unpatch_granite_for_dml() -> None:
    """Put the original forwards back."""
    for attr, original in list(_PATCHED.items()):
        for mod in _candidate_modules():
            cls = getattr(mod, attr, None)
            if cls is not None and getattr(cls.forward, "_usaf_sparse", False):
                cls.forward = original
                break
    _PATCHED.clear()
    _PATCHED_VALUES.clear()
