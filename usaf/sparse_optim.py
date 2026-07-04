import torch


class SparseAdam:
    """Adam that only updates active elements of each parameter.

    State (m, v) is stored on CPU for active elements only (via flat indices),
    not full-size tensors. Uses compact 1D parameter vectors aligned with
    ``active_idx`` when ``compact_params=True``.
    """

    def __init__(
        self,
        named_params: dict[str, torch.nn.Parameter],
        active_mask: dict[str, torch.Tensor] | None = None,
        active_idx: dict[str, torch.Tensor] | None = None,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        compact_params: bool = False,
    ):
        # compact_params: cada "param" é um vetor 1D só com os valores ATIVOS
        # (alinhado com active_idx), não o tensor full-size. Usado com
        # streaming quantizado + overlays, onde masters full-size (~4.8GB fp16)
        # não cabem na RAM. Requer step(compact_grads=...).
        self.compact_params = compact_params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self._named: dict[str, torch.nn.Parameter] = dict(named_params)
        self._step = 0

        self._active_ids: list[str] = []
        self._idx: dict[str, torch.Tensor] = {}       # flat active indices (CPU long)
        self._idx_dev: dict[str, torch.Tensor] = {}   # cached indices on param device
        self._m: dict[str, torch.Tensor] = {}         # momentum (CPU float32)
        self._v: dict[str, torch.Tensor] = {}

        if active_idx is not None:
            self._build_state_from_idx(active_idx)
        else:
            self._build_state(active_mask or {})

    def _build_state_from_idx(self, active_idx: dict[str, torch.Tensor]):
        """Build state from pre-computed flat indices (CPU long). Avoids full-size bool masks."""
        self._active_ids, self._idx, self._idx_dev, self._m, self._v = [], {}, {}, {}, {}
        for name, idx in active_idx.items():
            if name not in self._named or idx is None or idx.numel() == 0:
                continue
            idx = idx.reshape(-1).to(device="cpu", dtype=torch.long)
            self._active_ids.append(name)
            self._idx[name] = idx
            self._m[name] = torch.zeros(idx.numel(), device="cpu", dtype=torch.float32)
            self._v[name] = torch.zeros(idx.numel(), device="cpu", dtype=torch.float32)

    def _build_state(self, active_mask: dict[str, torch.Tensor]):
        self._active_ids = []
        self._idx, self._idx_dev, self._m, self._v = {}, {}, {}, {}
        for name, param in self._named.items():
            mask = active_mask.get(name)
            if mask is None:
                continue
            idx = mask.reshape(-1).to(device="cpu", dtype=torch.bool).nonzero(as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                continue
            self._active_ids.append(name)
            self._idx[name] = idx
            self._m[name] = torch.zeros(idx.numel(), device="cpu", dtype=torch.float32)
            self._v[name] = torch.zeros(idx.numel(), device="cpu", dtype=torch.float32)

    def _device_idx(self, name: str, param: torch.nn.Parameter) -> torch.Tensor:
        cached = self._idx_dev.get(name)
        if cached is None or cached.device != param.device:
            cached = self._idx[name].to(param.device)
            self._idx_dev[name] = cached
        return cached

    def zero_grad(self, set_to_none: bool = False):
        for name in self._active_ids:
            param = self._named[name]
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def step(self, compact_grads: dict[str, torch.Tensor] | None = None):
        """One Adam step over active elements.

        Args:
            compact_grads: optional pre-compacted grads (name -> 1D CPU tensor
                aligned with ``active_idx``, e.g. from ``SparseGradStore.compact``).
                Avoids materializing full-size grads for ephemeral streaming params.
        """
        self._step += 1
        beta1, beta2 = self.betas
        bias1 = 1 - beta1 ** self._step
        bias2 = 1 - beta2 ** self._step

        for name in self._active_ids:
            param = self._named[name]
            if compact_grads is None and param.grad is None:
                continue

            if compact_grads is not None:
                g = compact_grads.get(name)
                if g is None:
                    continue
                grad = g.detach().to(device="cpu", dtype=torch.float32)
            else:
                idx_dev = self._device_idx(name, param)
                # extrai só os gradientes ativos e traz pra CPU (estado vive em CPU)
                grad = param.grad.detach().reshape(-1).index_select(0, idx_dev).cpu().float()

            if self.weight_decay > 0:
                if self.compact_params:
                    p_active = param.detach().cpu().float()
                else:
                    idx_dev = self._device_idx(name, param)
                    p_active = param.detach().reshape(-1).index_select(0, idx_dev).cpu().float()
                grad = grad + self.weight_decay * p_active

            m = self._m[name]
            v = self._v[name]
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            update = (m / bias1) / ((v / bias2).sqrt() + self.eps)
            delta = (-self.lr * update).to(device=param.device, dtype=param.dtype)

            if self.compact_params:
                param.data.add_(delta)
            else:
                idx_dev = self._device_idx(name, param)
                flat = param.data.reshape(-1)
                flat.scatter_(0, idx_dev, flat.gather(0, idx_dev) + delta)

    def refresh(self, model):
        """Re-bind current model Parameters by name (needed for streaming experts).

        Expert hooks reassign ``module._parameters[name]`` each fwd/bwd cycle,
        so stored references become stale. The sparse state (idx/m/v) is indexed
        by name and stays valid; we only re-point to the live objects.
        """
        current = dict(model.named_parameters())
        for name in self._active_ids:
            if name not in current:
                continue
            if current[name].numel() != self._named[name].numel():
                continue
            self._named[name] = current[name]
        self._idx_dev = {}  # device pode ter mudado (expert volta pra CPU)

    def reselect(
        self,
        named_params: dict[str, torch.nn.Parameter],
        new_active_mask: dict[str, torch.Tensor],
    ):
        self._named = dict(named_params)
        self._idx_dev = {}
        self._build_state(new_active_mask)

    def state_dict(self) -> dict:
        """State for checkpoint/resume (m/v/step; idx comes from active_idx)."""
        return {
            "step": self._step,
            "m": {n: t.clone() for n, t in self._m.items()},
            "v": {n: t.clone() for n, t in self._v.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self._step = int(sd["step"])
        for n, t in sd["m"].items():
            if n in self._m and t.numel() == self._m[n].numel():
                self._m[n] = t.clone().float()
        for n, t in sd["v"].items():
            if n in self._v and t.numel() == self._v[n].numel():
                self._v[n] = t.clone().float()

    @property
    def num_active_params(self) -> int:
        return sum(idx.numel() for idx in self._idx.values())

    @property
    def optimizer_memory_mb(self) -> float:
        total = 0
        for name in self._active_ids:
            total += self._m[name].numel() * 4
            total += self._v[name].numel() * 4
        return total / (1024 * 1024)
