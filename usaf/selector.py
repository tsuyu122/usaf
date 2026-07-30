import torch
import numpy as np
from typing import Optional, Dict


def _kth_largest_threshold(scores: Dict[str, torch.Tensor], k: int) -> float:
    cat = torch.cat([s.reshape(-1) for s in scores.values()])
    n = cat.numel()
    k = max(1, min(k, n))
    arr = cat.numpy()
    kth = n - k
    threshold = float(np.partition(arr, kth)[kth])
    del cat, arr
    return threshold


class TopKSelector:
    def __init__(self, k: int):
        self.k = k

    def select(self, scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not scores:
            return {}
        threshold = _kth_largest_threshold(scores, self.k)
        return {name: (s >= threshold) for name, s in scores.items()}


class ThresholdSelector:
    def __init__(self, percentile: float):
        self.percentile = percentile

    def select(self, scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cat = torch.cat([s.reshape(-1) for s in scores.values()])
        n = cat.numel()
        idx = min(max(int(n * self.percentile / 100.0), 0), n - 1)
        arr = cat.numpy()
        threshold = float(np.partition(arr, idx)[idx])
        del cat, arr
        return {name: (s >= threshold) for name, s in scores.items()}


class DynamicSelector:
    def __init__(self, initial_k: int, reselect_every_n_steps: int, selection: str = "topk"):
        self.initial_k = initial_k
        self.reselect_every_n_steps = reselect_every_n_steps
        self.selection = selection
        self._step_counter = 0
        self._active_mask: Dict[str, torch.Tensor] = {}

    def should_reselect(self) -> bool:
        self._step_counter += 1
        return self._step_counter % self.reselect_every_n_steps == 0

    def update_mask(
        self,
        scores: Dict[str, torch.Tensor],
        k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        k = k or self.initial_k
        if self.selection == "topk":
            selector = TopKSelector(k)
        else:
            selector = ThresholdSelector(99.98)
        self._active_mask = selector.select(scores)
        return self._active_mask

    @property
    def active_mask(self) -> Dict[str, torch.Tensor]:
        return self._active_mask
