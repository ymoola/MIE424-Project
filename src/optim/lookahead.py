from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer


class Lookahead(Optimizer):
    """
    Lookahead wrapper optimizer.

    It wraps any base PyTorch optimizer (e.g., SGD, Adam), performs fast
    updates with the base optimizer, and every k steps interpolates fast
    weights into slow weights:

        slow <- slow + alpha * (fast - slow)
        fast <- slow
    """

    def __init__(self, base_optimizer: Optimizer, k: int = 5, alpha: float = 0.5) -> None:
        if not isinstance(base_optimizer, Optimizer):
            raise TypeError("base_optimizer must be an instance of torch.optim.Optimizer")
        if k < 1:
            raise ValueError("k must be >= 1")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0

        defaults = dict(k=k, alpha=alpha, **base_optimizer.defaults)
        super().__init__(base_optimizer.param_groups, defaults)

        self.param_groups = self.base_optimizer.param_groups

        self.slow_weights = [
            p.detach().clone()
            for group in self.param_groups
            for p in group["params"]
        ]
        for slow in self.slow_weights:
            slow.requires_grad = False

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter % self.k != 0:
            return loss

        idx = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param is None:
                    idx += 1
                    continue
                slow = self.slow_weights[idx]
                slow.add_(param.data - slow, alpha=self.alpha)
                param.data.copy_(slow)
                idx += 1

        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "slow_weights": [w.clone() for w in self.slow_weights],
            "k": self.k,
            "alpha": self.alpha,
            "step_counter": self.step_counter,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.param_groups = self.base_optimizer.param_groups

        self.k = int(state_dict.get("k", self.k))
        self.alpha = float(state_dict.get("alpha", self.alpha))
        self.step_counter = int(state_dict.get("step_counter", 0))

        loaded_slow = state_dict.get("slow_weights", None)
        if loaded_slow is None:
            self.slow_weights = [
                p.detach().clone()
                for group in self.param_groups
                for p in group["params"]
            ]
            for slow in self.slow_weights:
                slow.requires_grad = False
            return

        if len(loaded_slow) != sum(len(g["params"]) for g in self.param_groups):
            raise ValueError("Mismatch between slow_weights and optimizer parameters.")

        self.slow_weights = [w.detach().clone() for w in loaded_slow]
        for slow in self.slow_weights:
            slow.requires_grad = False
