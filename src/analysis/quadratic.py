"""
Synthetic quadratic minimization for analyzing Lookahead vs base optimizers.

Uses the same elliptical bowl as scripts/animation.py:
    f(x, y) = 0.5 * (0.2 x^2 + 2 y^2) = 0.1 x^2 + y^2
True gradient: (0.2 x, 2 y).

Optional Gaussian noise on gradients emulates stochastic optimization so
variance across iterates / runs is non-trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer, SGD

from src.optim import Lookahead

OptimizerName = Literal[
    "sgd",
    "sgd_momentum",
    "adam",
    "lookahead_sgd",
    "lookahead_sgd_momentum",
    "lookahead_adam",
]


@dataclass
class QuadraticConfig:
    """Diagonal Hessian H = diag(a_x, a_y); loss = 0.5 * (a_x x^2 + a_y y^2)."""

    a_x: float = 0.2
    a_y: float = 2.0


def elliptical_quadratic_loss(theta: torch.Tensor, cfg: QuadraticConfig) -> torch.Tensor:
    x, y = theta[0], theta[1]
    return 0.5 * (cfg.a_x * x * x + cfg.a_y * y * y)


class _PointModel(nn.Module):
    """Single 2D parameter for synthetic quadratic optimization."""

    def __init__(self, init: torch.Tensor) -> None:
        super().__init__()
        self.theta = nn.Parameter(init.clone())


def build_quadratic_optimizer(
    name: OptimizerName,
    params,
    lr: float,
    momentum: float,
    weight_decay: float,
    lookahead_k: int,
    lookahead_alpha: float,
) -> Optimizer:
    name = name.lower()  # type: ignore[assignment]
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "lookahead_sgd":
        base = SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
        return Lookahead(base_optimizer=base, k=lookahead_k, alpha=lookahead_alpha)
    if name == "lookahead_sgd_momentum":
        base = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        return Lookahead(base_optimizer=base, k=lookahead_k, alpha=lookahead_alpha)
    if name == "lookahead_adam":
        base = Adam(params, lr=lr, weight_decay=weight_decay)
        return Lookahead(base_optimizer=base, k=lookahead_k, alpha=lookahead_alpha)
    raise ValueError(f"Unknown optimizer name: {name}")


def run_quadratic_trajectory(
    optimizer_name: OptimizerName,
    cfg: QuadraticConfig,
    init: torch.Tensor,
    noise: torch.Tensor,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    lookahead_k: int,
    lookahead_alpha: float,
    sigma: float,
    num_steps: int,
    device: torch.device,
    record_slow_on_sync: bool = True,
) -> dict[str, Any]:
    """
    Run one trajectory with a fixed noise schedule noise[t] shape (num_steps, 2).

    Returns dict with:
      - trajectory: (T+1, 2) positions after each step (fast / effective params)
      - losses: (T+1,) loss at each recorded point (before step 0 = init)
      - dist_to_opt: (T+1,) Euclidean distance to origin
      - slow_snapshots: list of (step_idx, tensor) when Lookahead syncs (else empty)
      - fast_before_sync: fast weights immediately before each Lookahead sync (else empty)
    """
    init = init.to(device)
    noise = noise.to(device)
    model = _PointModel(init).to(device)
    opt = build_quadratic_optimizer(
        optimizer_name,
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        lookahead_k=lookahead_k,
        lookahead_alpha=lookahead_alpha,
    )

    traj: list[torch.Tensor] = []
    losses: list[float] = []
    dists: list[float] = []
    slow_snaps: list[tuple[int, torch.Tensor]] = []
    fast_before_sync: list[torch.Tensor] = []

    theta = model.theta
    with torch.no_grad():
        traj.append(theta.detach().clone())
        losses.append(elliptical_quadratic_loss(theta, cfg).item())
        dists.append(theta.norm().item())

    for t in range(num_steps):
        if isinstance(opt, Lookahead) and (t + 1) % opt.k == 0:
            fast_before_sync.append(theta.detach().clone().cpu())

        opt.zero_grad(set_to_none=True)
        loss = elliptical_quadratic_loss(theta, cfg)
        loss.backward()
        with torch.no_grad():
            if sigma > 0:
                theta.grad.add_(sigma * noise[t])
        opt.step()

        if record_slow_on_sync and isinstance(opt, Lookahead) and opt.sync_happened:
            slow = torch.cat([w.view(-1) for w in opt.slow_weights])
            slow_snaps.append((t + 1, slow.detach().cpu()))

        with torch.no_grad():
            traj.append(theta.detach().clone())
            losses.append(elliptical_quadratic_loss(theta, cfg).item())
            dists.append(theta.norm().item())

    return {
        "trajectory": torch.stack(traj).cpu(),
        "losses": torch.tensor(losses, dtype=torch.float32),
        "dist_to_opt": torch.tensor(dists, dtype=torch.float32),
        "slow_snapshots": slow_snaps,
        "fast_before_sync": fast_before_sync,
    }


def variance_across_runs(
    optimizer_name: OptimizerName,
    cfg: QuadraticConfig,
    *,
    init: torch.Tensor,
    lr: float,
    momentum: float,
    weight_decay: float,
    lookahead_k: int,
    lookahead_alpha: float,
    sigma: float,
    num_steps: int,
    num_runs: int,
    base_seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Independent noise streams per run; return variance of final distance and final loss."""
    finals_dist: list[float] = []
    finals_loss: list[float] = []
    for r in range(num_runs):
        g = torch.Generator(device=device)
        g.manual_seed(base_seed + r)
        noise = torch.randn(num_steps, 2, device=device, generator=g)
        out = run_quadratic_trajectory(
            optimizer_name,
            cfg,
            init,
            noise,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            lookahead_k=lookahead_k,
            lookahead_alpha=lookahead_alpha,
            sigma=sigma,
            num_steps=num_steps,
            device=device,
            record_slow_on_sync=False,
        )
        finals_dist.append(out["dist_to_opt"][-1].item())
        finals_loss.append(out["losses"][-1].item())
    d = torch.tensor(finals_dist, dtype=torch.float64)
    ell = torch.tensor(finals_loss, dtype=torch.float64)
    return {
        "var_final_dist": float(d.var(unbiased=False)),
        "mean_final_dist": float(d.mean()),
        "var_final_loss": float(ell.var(unbiased=False)),
        "mean_final_loss": float(ell.mean()),
    }


def fast_slow_variance_from_snapshots(
    lookahead_snapshots: list[tuple[int, torch.Tensor]],
    fast_at_sync: list[torch.Tensor],
) -> tuple[float, float] | None:
    """
    Compare variance of fast vs slow positions at sync indices (2D coords).
    fast_at_sync[i] should align with snapshots[i][1].
    """
    if not lookahead_snapshots or not fast_at_sync:
        return None
    slow = torch.stack([s[1].float() for s in lookahead_snapshots])
    fast = torch.stack([f.float() for f in fast_at_sync])
    if slow.shape != fast.shape:
        return None
    var_slow = slow.var(dim=0, unbiased=False).sum().item()
    var_fast = fast.var(dim=0, unbiased=False).sum().item()
    return var_slow, var_fast
