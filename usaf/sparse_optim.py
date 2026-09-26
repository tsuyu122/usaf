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
        self.compact_params = compact_params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self._named: dict[str, torch.nn.Parameter] = dict(named_params)
        self._step = 0

        self._active_ids: list[str] = []
        self._idx: dict[str, torch.Tensor] = {}
        self._idx_dev: dict[str, torch.Tensor] = {}
        self._m: dict[str, torch.Tensor] = {}
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
                # contiguous(), not reshape(-1): reshape copies when the
                # tensor is not contiguous, and the scatter_ that follows then
                # writes the Adam delta into a temporary. The parameter comes
                # out bit-identical - no error, no warning - while the router
                # params still move, so the loss still falls. contiguous()
                # keeps the shape (an expert is [E, H, I], and flattening the
                # adopted copy would feed a 1-D weight to the next forward) and
                # makes the view() below a guaranteed view rather than a copy.
                if not param.data.is_contiguous():
                    param.data = param.data.contiguous()
                flat = param.data.view(-1)
                flat.scatter_(0, idx_dev, flat.gather(0, idx_dev) + delta)

    def refresh(self, model) -> None:
        """Re-bind current model Parameters by name (streaming experts).

        Expert hooks reassign ``module._parameters[name]`` on every
        fwd/bwd cycle, so the references captured at construction go stale
        after the first backward. The sparse state (idx/m/v) is keyed by name
        and stays valid, so only the Parameter objects are re-pointed.

        A tensor whose element count differs is skipped on purpose: with
        ``compact_params=True`` the optimizer owns compact 1D masters whose
        size deliberately differs from the streamed expert tensor.
        """
        current = dict(model.named_parameters())
        for name in self._active_ids:
            new = current.get(name)
            if new is None or new.numel() != self._named[name].numel():
                continue
            self._named[name] = new
        self._idx_dev = {}

    def reselect(
        self,
        named_params: dict[str, torch.nn.Parameter],
        new_active_idx: dict[str, torch.Tensor],
    ) -> None:
        """Rebind to a new active set after a RigL reselection.

        Adam moments for elements that stay active are carried over, so the
        optimizer state is not reset every reselection. The previous version
        called _build_state(), which reallocated m and v to zero and kept the
        old step count - every reselection therefore wiped the momentum and
        desynchronised the bias correction from the moment tensors.

        Args:
            named_params: name -> live Parameter for each active tensor.
            new_active_idx: name -> 1D CPU long tensor of active flat indices.
        """
        prev_m, prev_v, prev_idx = self._m, self._v, self._idx
        self._named = dict(named_params)
        self._build_state_from_idx(new_active_idx)

        for name in self._active_ids:
            old_idx, new_i = prev_idx.get(name), self._idx.get(name)
            if old_idx is None or new_i is None:
                continue
            # Map each new active index back to its position in the old set so
            # the surviving elements keep their m/v. The lookup must be sized
            # by the larger of the two index ranges, because a newly activated
            # element can sit beyond every previously active index.
            span = int(max(old_idx.max().item(), new_i.max().item())) + 1
            lookup = torch.full((span,), -1, dtype=torch.long)
            lookup[old_idx] = torch.arange(old_idx.numel(), dtype=torch.long)
            pos = lookup[new_i]
            keep = pos >= 0
            if not bool(keep.any()):
                continue
            self._m[name][keep] = prev_m[name][pos[keep]]
            self._v[name][keep] = prev_v[name][pos[keep]]

    def state_dict(self) -> dict:
        """State for checkpoint/resume (m/v/step; idx comes from active_idx)."""
        return {
            "step": self._step,
            "m": {n: t.clone() for n, t in self._m.items()},
            "v": {n: t.clone() for n, t in self._v.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore optimizer state from a checkpoint.

        The previous version skipped any tensor whose element count did not
        match and reported nothing, so a checkpoint whose active set no longer
        lines up with the current one restored the step counter while leaving
        m and v at zero - Adam then divided by a bias correction derived from
        a step count it had no moments for. Mismatches are now reported.
        """
        self._step = int(sd["step"])
        skipped: list[str] = []
        for field, store in (("m", self._m), ("v", self._v)):
            for n, t in sd.get(field, {}).items():
                if n not in store:
                    skipped.append(f"{field}:{n} (not active now)")
                    continue
                if t.numel() != store[n].numel():
                    skipped.append(
                        f"{field}:{n} ({t.numel()} vs {store[n].numel()} elements)"
                    )
                    continue
                store[n] = t.clone().float()
        if skipped:
            raise ValueError(
                "optimizer state does not match the active set; "
                + f"{len(skipped)} tensor(s) skipped: "
                + ", ".join(skipped[:5])
                + (" ..." if len(skipped) > 5 else "")
            )

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
