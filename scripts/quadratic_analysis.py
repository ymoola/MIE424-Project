"""
Quadratic analysis: variance reduction and convergence of the Lookahead optimizer.

Reproduces the theoretical analysis from Section 4 of:
  "Lookahead Optimizer: k steps forward, 1 step back" (Zhang et al., 2019)

on the 2-D quadratic loss  f(x, y) = 0.1*x^2 + y^2,
which is the same loss landscape used in animation.py.

Outputs (saved to results/figures/quadratic/):
  1. trajectory.png    -- 2D contour plot with optimizer paths
  2. convergence.png   -- distance to optimum vs step (log scale)
  3. loss_curve.png    -- loss value vs step (log scale)
  4. variance.png      -- variance comparison: base vs fast vs slow weights

Run from the project root:
    python scripts/quadratic_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.quadratic import (
    QuadraticProblem,
    SGDOptimizer,
    SGDMomentumOptimizer,
    AdamOptimizer,
    run_optimizer,
    run_lookahead,
    distance_to_optimum,
    trajectory_variance,
    variance_summary,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

START       = np.array([-4.0, 2.0])   # same starting point as animation.py
N_STEPS     = 150
K           = 5                        # Lookahead inner steps
ALPHA       = 0.5                      # Lookahead interpolation factor

# Learning rates tuned so SGD visibly oscillates but converges
LR_SGD      = 0.85   # large enough to zigzag along steep y-axis
LR_MOMENTUM = 0.40
LR_ADAM     = 0.30

OUT_DIR = PROJECT_ROOT / "results" / "figures" / "quadratic"

# ---------------------------------------------------------------------------
# Colour palette (consistent across all figures)
# ---------------------------------------------------------------------------
COLORS = {
    "sgd":               "#e41a1c",   # red
    "sgd_momentum":      "#ff7f00",   # orange
    "adam":              "#4daf4a",   # green
    "la_sgd_fast":       "#377eb8",   # blue (dashed/thin)
    "la_sgd_slow":       "#377eb8",   # blue (thick)
    "la_adam_fast":      "#984ea3",   # purple (dashed/thin)
    "la_adam_slow":      "#984ea3",   # purple (thick)
}


# ---------------------------------------------------------------------------
# Run all optimizers
# ---------------------------------------------------------------------------

def _run_all(problem: QuadraticProblem) -> tuple:
    sgd_res     = run_optimizer(SGDOptimizer(LR_SGD),              START, N_STEPS, problem)
    mom_res     = run_optimizer(SGDMomentumOptimizer(LR_MOMENTUM), START, N_STEPS, problem)
    adam_res    = run_optimizer(AdamOptimizer(LR_ADAM),            START, N_STEPS, problem)

    la_sgd_res  = run_lookahead(SGDOptimizer(LR_SGD),              START, N_STEPS, problem, K, ALPHA)
    la_adam_res = run_lookahead(AdamOptimizer(LR_ADAM),            START, N_STEPS, problem, K, ALPHA)

    return sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res


# ---------------------------------------------------------------------------
# Figure 1 — Trajectory on the 2D loss landscape
# ---------------------------------------------------------------------------

def _fig_trajectory(
    problem: QuadraticProblem,
    sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        r"Optimizer Trajectories on $f(x,y) = 0.1x^2 + y^2$",
        fontsize=14, fontweight="bold",
    )

    # Shared contour grid
    xs = np.linspace(-5, 5, 400)
    ys = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.1 * X**2 + Y**2
    levels = [0.05, 0.2, 0.5, 1, 2, 4, 8]

    for ax, title, show_lookahead in [
        (axes[0], "Base Optimizers", False),
        (axes[1], "Base vs Lookahead Slow Weights", True),
    ]:
        ax.contour(X, Y, Z, levels=levels, colors="grey", alpha=0.35, linewidths=0.8)
        ax.set_xlim(-5, 1)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.axvline(0, color="k", lw=0.4, ls="--")
        ax.plot(*problem.optimum, "k*", markersize=12, label="Optimum (0,0)")

        def _plot_traj(traj, color, label, lw=1.5, ls="-", alpha=0.9, zorder=3):
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=lw,
                    ls=ls, alpha=alpha, zorder=zorder, label=label)
            ax.plot(*traj[0], "o", color=color, markersize=5, zorder=4)

        _plot_traj(sgd_res["trajectory"],   COLORS["sgd"],          "SGD",           lw=1.8)
        _plot_traj(mom_res["trajectory"],   COLORS["sgd_momentum"], "SGD+Momentum",  lw=1.8)
        _plot_traj(adam_res["trajectory"],  COLORS["adam"],         "Adam",          lw=1.8)

        if show_lookahead:
            _plot_traj(
                la_sgd_res["fast_trajectory"],
                COLORS["la_sgd_fast"],
                "Lookahead-SGD [fast]", lw=0.9, ls="--", alpha=0.5,
            )
            _plot_traj(
                la_sgd_res["slow_trajectory"],
                COLORS["la_sgd_slow"],
                "Lookahead-SGD [slow]", lw=2.5, zorder=5,
            )
            _plot_traj(
                la_adam_res["slow_trajectory"],
                COLORS["la_adam_slow"],
                "Lookahead-Adam [slow]", lw=2.5, zorder=5,
            )

        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 2 — Convergence: distance to optimum
# ---------------------------------------------------------------------------

def _fig_convergence(
    problem: QuadraticProblem,
    sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = np.arange(N_STEPS + 1)

    def _d(traj):
        return distance_to_optimum(traj, problem.optimum)

    ax.semilogy(steps, _d(sgd_res["trajectory"]),
                color=COLORS["sgd"], lw=1.8, label="SGD")
    ax.semilogy(steps, _d(mom_res["trajectory"]),
                color=COLORS["sgd_momentum"], lw=1.8, label="SGD+Momentum")
    ax.semilogy(steps, _d(adam_res["trajectory"]),
                color=COLORS["adam"], lw=1.8, label="Adam")
    ax.semilogy(steps, _d(la_sgd_res["fast_trajectory"]),
                color=COLORS["la_sgd_fast"], lw=1.0, ls="--", alpha=0.6,
                label="Lookahead-SGD [fast]")
    ax.semilogy(steps, _d(la_sgd_res["slow_trajectory"]),
                color=COLORS["la_sgd_slow"], lw=2.5, label="Lookahead-SGD [slow]")
    ax.semilogy(steps, _d(la_adam_res["slow_trajectory"]),
                color=COLORS["la_adam_slow"], lw=2.5, label="Lookahead-Adam [slow]")

    # Mark sync points
    sync_steps = [i for i in range(K, N_STEPS + 1, K)]
    for s in sync_steps:
        ax.axvline(s, color="grey", lw=0.4, ls=":", alpha=0.5)

    ax.set_xlabel("Gradient step", fontsize=12)
    ax.set_ylabel(r"$\|\theta_t - \theta^*\|$  (log scale)", fontsize=12)
    ax.set_title(
        f"Convergence to Optimum  (k={K}, α={ALPHA})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 3 — Loss curves
# ---------------------------------------------------------------------------

def _fig_loss(
    sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = np.arange(N_STEPS + 1)

    ax.semilogy(steps, sgd_res["losses"],
                color=COLORS["sgd"], lw=1.8, label="SGD")
    ax.semilogy(steps, mom_res["losses"],
                color=COLORS["sgd_momentum"], lw=1.8, label="SGD+Momentum")
    ax.semilogy(steps, adam_res["losses"],
                color=COLORS["adam"], lw=1.8, label="Adam")
    ax.semilogy(steps, la_sgd_res["fast_losses"],
                color=COLORS["la_sgd_fast"], lw=1.0, ls="--", alpha=0.6,
                label="Lookahead-SGD [fast]")
    ax.semilogy(steps, la_sgd_res["slow_losses"],
                color=COLORS["la_sgd_slow"], lw=2.5, label="Lookahead-SGD [slow]")
    ax.semilogy(steps, la_adam_res["slow_losses"],
                color=COLORS["la_adam_slow"], lw=2.5, label="Lookahead-Adam [slow]")

    ax.set_xlabel("Gradient step", fontsize=12)
    ax.set_ylabel(r"$f(\theta_t)$  (log scale)", fontsize=12)
    ax.set_title(
        r"Loss Curves on $f(x,y) = 0.1x^2 + y^2$"
        f"  (k={K}, α={ALPHA})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 4 — Variance analysis
# ---------------------------------------------------------------------------

def _fig_variance(
    sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
    out_path: Path,
) -> None:
    # Compute summaries
    la_sgd_summary  = variance_summary(sgd_res,  la_sgd_res)
    la_adam_summary = variance_summary(adam_res, la_adam_res)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f"Variance Reduction by Lookahead  (α={ALPHA}, k={K})\n"
        r"Theory predicts: Var(slow) $\approx \alpha^2 \cdot$ Var(fast) = "
        f"{ALPHA**2:.2f} × Var(fast)",
        fontsize=12, fontweight="bold",
    )

    for ax, summary, base_res, base_color, la_fast_color, la_slow_color, title in [
        (
            axes[0], la_sgd_summary, sgd_res,
            COLORS["sgd"], COLORS["la_sgd_fast"], COLORS["la_sgd_slow"],
            "SGD  vs  Lookahead-SGD",
        ),
        (
            axes[1], la_adam_summary, adam_res,
            COLORS["adam"], COLORS["la_adam_fast"], COLORS["la_adam_slow"],
            "Adam  vs  Lookahead-Adam",
        ),
    ]:
        labels = ["Base\noptimizer", "Lookahead\n[fast weights]", "Lookahead\n[slow weights]"]
        values = [summary["base_variance"], summary["fast_variance"], summary["slow_variance"]]
        colors = [base_color, la_fast_color, la_slow_color]
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.6)

        # Annotate bars with exact values
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.03,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        # Annotate empirical vs theoretical ratio
        emp_ratio  = summary["empirical_ratio"]
        theo_ratio = summary["theoretical_ratio"]
        ax.text(
            0.97, 0.97,
            f"Empirical  ratio: {emp_ratio:.3f}\n"
            f"Theoretical ratio: {theo_ratio:.3f}  (α²)",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"),
        )

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Trajectory variance  (sum of dim variances)", fontsize=9)
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 5 — Rolling variance over time (shows stabilisation)
# ---------------------------------------------------------------------------

def _fig_rolling_variance(
    sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
    out_path: Path,
    window: int = 20,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Rolling Variance of Iterates  (window={window} steps)\n"
        "Lower = more stable updates",
        fontsize=12, fontweight="bold",
    )

    def _rolling_var(traj: np.ndarray) -> np.ndarray:
        out = np.full(len(traj), np.nan)
        for i in range(window, len(traj)):
            out[i] = np.var(traj[i - window:i], axis=0).sum()
        return out

    steps = np.arange(N_STEPS + 1)

    for ax, base_res, base_color, la_res, la_fast_color, la_slow_color, title in [
        (
            axes[0], sgd_res, COLORS["sgd"], la_sgd_res,
            COLORS["la_sgd_fast"], COLORS["la_sgd_slow"],
            "SGD  vs  Lookahead-SGD",
        ),
        (
            axes[1], adam_res, COLORS["adam"], la_adam_res,
            COLORS["la_adam_fast"], COLORS["la_adam_slow"],
            "Adam  vs  Lookahead-Adam",
        ),
    ]:
        ax.plot(steps, _rolling_var(base_res["trajectory"]),
                color=base_color, lw=1.8, label="Base optimizer")
        ax.plot(steps, _rolling_var(la_res["fast_trajectory"]),
                color=la_fast_color, lw=1.0, ls="--", alpha=0.7, label="Lookahead [fast]")
        ax.plot(steps, _rolling_var(la_res["slow_trajectory"]),
                color=la_slow_color, lw=2.5, label="Lookahead [slow]")

        # Shade the α² theoretical band
        base_rv = _rolling_var(base_res["trajectory"])
        valid = ~np.isnan(base_rv)
        ax.fill_between(
            steps[valid],
            np.zeros(valid.sum()),
            base_rv[valid] * (ALPHA ** 2),
            color=la_slow_color, alpha=0.12,
            label=f"Theoretical target (α²·base = {ALPHA**2:.2f}·base)",
        )

        ax.set_xlabel("Gradient step", fontsize=11)
        ax.set_ylabel("Rolling variance", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Summary table printed to stdout
# ---------------------------------------------------------------------------

def _print_summary(
    sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
    problem: QuadraticProblem,
) -> None:
    la_sgd_summary  = variance_summary(sgd_res,  la_sgd_res)
    la_adam_summary = variance_summary(adam_res, la_adam_res)

    def _final_dist(traj):
        return float(distance_to_optimum(traj[-1:], problem.optimum)[0])

    def _final_loss(losses):
        return float(losses[-1])

    rows = [
        ("SGD",
         sgd_res["losses"][0], _final_loss(sgd_res["losses"]),
         _final_dist(sgd_res["trajectory"]),
         trajectory_variance(sgd_res["trajectory"]), "-"),
        ("SGD+Momentum",
         mom_res["losses"][0], _final_loss(mom_res["losses"]),
         _final_dist(mom_res["trajectory"]),
         trajectory_variance(mom_res["trajectory"]), "-"),
        ("Adam",
         adam_res["losses"][0], _final_loss(adam_res["losses"]),
         _final_dist(adam_res["trajectory"]),
         trajectory_variance(adam_res["trajectory"]), "-"),
        ("Lookahead-SGD [fast]",
         la_sgd_res["fast_losses"][0], _final_loss(la_sgd_res["fast_losses"]),
         _final_dist(la_sgd_res["fast_trajectory"]),
         la_sgd_summary["fast_variance"],
         f'{la_sgd_summary["empirical_ratio"]:.3f} (theory {ALPHA**2:.3f})'),
        ("Lookahead-SGD [slow]",
         la_sgd_res["slow_losses"][0], _final_loss(la_sgd_res["slow_losses"]),
         _final_dist(la_sgd_res["slow_trajectory"]),
         la_sgd_summary["slow_variance"],
         f'{la_sgd_summary["empirical_ratio"]:.3f} (theory {ALPHA**2:.3f})'),
        ("Lookahead-Adam [fast]",
         la_adam_res["fast_losses"][0], _final_loss(la_adam_res["fast_losses"]),
         _final_dist(la_adam_res["fast_trajectory"]),
         la_adam_summary["fast_variance"],
         f'{la_adam_summary["empirical_ratio"]:.3f} (theory {ALPHA**2:.3f})'),
        ("Lookahead-Adam [slow]",
         la_adam_res["slow_losses"][0], _final_loss(la_adam_res["slow_losses"]),
         _final_dist(la_adam_res["slow_trajectory"]),
         la_adam_summary["slow_variance"],
         f'{la_adam_summary["empirical_ratio"]:.3f} (theory {ALPHA**2:.3f})'),
    ]

    header = f"{'Optimizer':<30} {'Init loss':>10} {'Final loss':>12} {'Dist to opt':>12} {'Variance':>10} {'Var ratio (slow/fast)':>28}"
    print("\n" + "=" * len(header))
    print("QUADRATIC ANALYSIS SUMMARY")
    print(f"f(x,y) = 0.1x² + y²   |   start={START}   |   steps={N_STEPS}   |   k={K}, α={ALPHA}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, init_l, final_l, dist, var, ratio in rows:
        print(f"{name:<30} {init_l:>10.4f} {final_l:>12.6f} {dist:>12.6f} {var:>10.4f} {ratio:>28}")
    print("=" * len(header))
    print(f"\nTheoretical variance reduction factor: α² = {ALPHA}² = {ALPHA**2}")
    print("Slow weights should have ~α² × the variance of fast weights.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = QuadraticProblem()

    print(f"Running quadratic analysis on f(x,y) = 0.1x² + y²")
    print(f"  Start: {START},  Steps: {N_STEPS},  k={K},  α={ALPHA}")
    print(f"  LR: SGD={LR_SGD}, SGD+Mom={LR_MOMENTUM}, Adam={LR_ADAM}")
    print()

    sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res = _run_all(problem)

    print("Generating figures...")
    _fig_trajectory(problem, sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
                    OUT_DIR / "trajectory.png")
    _fig_convergence(problem, sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
                     OUT_DIR / "convergence.png")
    _fig_loss(sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
              OUT_DIR / "loss_curve.png")
    _fig_variance(sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
                  OUT_DIR / "variance.png")
    _fig_rolling_variance(sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res,
                          OUT_DIR / "rolling_variance.png")

    _print_summary(sgd_res, mom_res, adam_res, la_sgd_res, la_adam_res, problem)
    print(f"All figures saved to: {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
