import torch
from collections import OrderedDict


class ActivationCache:
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self._cache: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        self._invalidated_modules: set[str] = set()
        self._hooks: list = []
        self._step: int = 0

    def register_hooks(self, model: torch.nn.Module):
        self._hooks = []
        for name, module in model.named_modules():
            if self._is_transformer_block(module):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)
                self._cache[name] = []

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
                self._cache[name] = []
            self._cache[name].append(cached)
            return output
        return hook

    def get_cached(self, name: str, step: int) -> torch.Tensor:
        if name in self._cache and step < len(self._cache[name]):
            cached = self._cache[name][step]
            if isinstance(cached, torch.Tensor):
                return cached.to(self.device)
            return cached
        return None

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
        total = 0
        for name, tensors in self._cache.items():
            for t in tensors:
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        return total / (1024 * 1024)
