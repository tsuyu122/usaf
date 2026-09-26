from collections import OrderedDict

import torch


class ActivationCache:
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        # name -> step -> the activations captured for that step.
        self._cache: OrderedDict[str, dict[int, list]] = OrderedDict()
        self._invalidated_modules: set[str] = set()
        self._hooks: list = []
        self._step: int = 0
        # Which modules were called more than once inside some step. Kept so the
        # ambiguity is a thing a caller can ask about rather than guess at.
        self._ambiguous: dict[str, set[bool]] = {}

    def register_hooks(self, model: torch.nn.Module):
        self._hooks = []
        for name, module in model.named_modules():
            if self._is_transformer_block(module):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)
                self._cache[name] = {}

    def _is_transformer_block(self, module: torch.nn.Module) -> bool:
        cls_name = module.__class__.__name__
        block_names = [
            "DecoderLayer",
            "TransformerBlock",
            "GemmaDecoderLayer",
            "Gemma4DecoderLayer",
            "Block",
        ]
        return any(bn in cls_name for bn in block_names)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if name in self._invalidated_modules:
                return output
            if isinstance(output, tuple):
                cached = tuple(
                    t.detach().to(self.device) if isinstance(t, torch.Tensor) else t
                    for t in output
                )
            elif isinstance(output, torch.Tensor):
                cached = output.detach().to(self.device)
            else:
                cached = output
            if name not in self._cache:
                self._cache[name] = {}
            # Keyed by step, not appended to a list and later indexed by step.
            # Those are the same thing only while every module is called exactly
            # once per step, and a module called twice - which a reused decoder
            # layer, a second pass, or anything wrapping the block does - shifts
            # every later index by one. The result is not a crash. It is a cache
            # that confidently returns the previous step's activation.
            per_step = self._cache[name].setdefault(self._step, [])
            per_step.append(cached)
            self._ambiguous.setdefault(name, set()).add(len(per_step) > 1)
            return output
        return hook

    def get_cached(self, name: str, step: int) -> torch.Tensor:
        """The activation captured for *step*, or None if it was not captured.

        A step with no entry returns None rather than the nearest one it could
        find, because a stale activation read back as a fresh one is the one
        failure this class cannot report.
        """
        if step < 0:
            raise ValueError(
                f"step must not be negative, got {step}; a negative index used "
                f"to read the most recent entry instead of failing"
            )
        per_step = self._cache.get(name)
        if not per_step:
            return None
        entries = per_step.get(step)
        if not entries:
            return None
        cached = entries[0]
        if isinstance(cached, torch.Tensor):
            return cached.to(self.device)
        return cached

    def calls_in_step(self, name: str, step: int) -> int:
        """How many times *name* was called during *step*."""
        per_step = self._cache.get(name)
        if not per_step:
            return 0
        return len(per_step.get(step, ()))

    def invalidate(self, module_names: set[str]):
        for name in module_names:
            self._invalidated_modules.add(name)
            if name in self._cache:
                del self._cache[name]

    def reset(self):
        self._cache.clear()
        self._invalidated_modules.clear()
        self._step = 0

    def advance_step(self):
        self._step += 1

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @property
    def cached_modules(self) -> set[str]:
        return set(self._cache.keys())

    def memory_estimate_mb(self) -> float:
        """Bytes held by the cache, walking into container entries.

        Transformer blocks return a tuple, and that is exactly what the hook
        stores, so counting only bare tensors reported zero for the modules the
        cache exists to hold - the estimate claimed the cache was free.
        """
        def _nbytes(obj) -> int:
            if isinstance(obj, torch.Tensor):
                return obj.numel() * obj.element_size()
            if isinstance(obj, (tuple, list)):
                return sum(_nbytes(o) for o in obj)
            if isinstance(obj, dict):
                return sum(_nbytes(o) for o in obj.values())
            return 0

        total = 0
        for steps in self._cache.values():
            for calls in steps.values():
                for entry in calls:
                    total += _nbytes(entry)
        return total / (1024 * 1024)
