import torch
from torch import nn

import copy

from typing import Optional


def _unwrap(module: nn.Module) -> nn.Module:
    """torch.compile wraps modules in an OptimizedModule; return the plain module."""
    return getattr(module, "_orig_mod", module)


class EMAHelper:
    """
    Exponential moving average of a model's weights.

    The helper owns `self.model`: a frozen, eval-mode copy of the registered module
    whose parameters ARE the running average. `update()` nudges them towards the live
    weights after each optimizer step, so `self.model` is always ready to use for
    validation, sampling/evaluation and saving.

    Parameters are averaged. Buffers (e.g. BatchNorm running stats) are copied from
    the live model, since they are not trained.
    """

    def __init__(self, mu=0.999, device="cpu"):
        self.mu = mu
        self.device = device
        self.step = 0  # Track total updates for dynamic warmup
        self.model: Optional[nn.Module] = None
        self._ema_params: list = []
        self._ema_buffers: list = []

    @torch.no_grad()
    def register(self, module: nn.Module) -> None:
        module = _unwrap(module)
        self.model = copy.deepcopy(module).to(self.device).eval().requires_grad_(False)
        self._ema_params = list(self.model.parameters())
        self._ema_buffers = list(self.model.buffers())
        self.step = 0

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        module = _unwrap(module)
        self.step += 1

        # Warmup: early on the average would mostly contain the random init, so use a
        # short window first and let it grow towards mu.
        decay = min(self.mu, (1.0 + self.step) / (10.0 + self.step))

        # ema <- ema + (1 - decay) * (live - ema), fused over all tensors
        torch._foreach_lerp_(self._ema_params, list(module.parameters()), 1.0 - decay)

        # Buffers are not trained: just mirror them (empty loop for GroupNorm/LayerNorm nets)
        for b_ema, b in zip(self._ema_buffers, module.buffers()):
            b_ema.copy_(b)

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> nn.Module:
        """Load the EMA weights into another module (same architecture), in place."""
        module = _unwrap(module)
        for dst, src in zip(module.parameters(), self._ema_params):
            dst.copy_(src)
        for dst, src in zip(module.buffers(), self._ema_buffers):
            dst.copy_(src)
        return module
