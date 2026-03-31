"""
Run synthetic quadratic analysis (Lookahead paper style) and save figures + CSV.

Outputs under results/quadratic/:
  - trajectories.png
  - distance_to_optimum.png
  - variance_across_runs.png
  - lookahead_fast_vs_slow_variance.png  (temporal variance at sync points)
  - metrics.csv
  - README snippet: theory note about alpha^2 variance reduction
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Non-interactive backend for headless / CI
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.quadratic import (
    QuadraticConfig,
    OptimizerName,
    elliptical_quadratic_loss,
    run_quadratic_trajectory,
    variance_across_runs,
)


def _contour_axes(cfg: QuadraticConfig, lim: float = 5.0):
    xs = np.linspace(-lim, lim, 200)
    ys = np.linspace(-lim, lim, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (cfg.a_x * X**2 + cfg.a_y * Y**2)
    return X, Y, Z


def main() -> None:
    parser = argparse.ArgumentParser(description="Quadratic bowl analysis for Lookahead.")
    parser.add_argument("--out-dir", type=str, default="results/quadratic")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--sigma", type=float, default=0.08, help="Gradient noise std.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shared noise schedule.")
    parser.add_argument("--var-runs", type=int, default=64, help="Runs for variance bar chart.")
    parser.add_argument("--lr-sgd", type=float, default=0.12)
    parser.add_argument("--lr-adam", type=float, default=0.08)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lookahead-k", type=int, default=5)
    parser.add_argument("--lookahead-alpha", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cpu")
    cfg = QuadraticConfig()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init = torch.tensor([-4.0, 2.0], dtype=torch.float32)
    torch.manual_seed(args.seed)
    shared_noise = torch.randn(args.steps, 2, device=device)

    optimizers: list[OptimizerName] = [
        "sgd",
        "sgd_momentum",
        "adam",
        "lookahead_sgd_momentum",
        "lookahead_adam",
    ]

    def lr_for(name: str) -> float:
        return args.lr_adam if "adam" in name else args.lr_sgd

    trajectories: dict[str, torch.Tensor] = {}
    dists: dict[str, torch.Tensor] = {}

    for name in optimizers:
        out = run_quadratic_trajectory(
            name,
            cfg,
            init,
            shared_noise,
            lr=lr_for(name),
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            lookahead_k=args.lookahead_k,
            lookahead_alpha=args.lookahead_alpha,
            sigma=args.sigma,
            num_steps=args.steps,
            device=device,
            record_slow_on_sync=True,
        )
        trajectories[name] = out["trajectory"]
        dists[name] = out["dist_to_opt"]

    # --- Figure 1: trajectories on loss landscape ---
    fig, ax = plt.subplots(figsize=(8, 7))
    X, Y, Z = _contour_axes(cfg)
    ax.contour(X, Y, Z, levels=np.logspace(-2, 2, 12), colors="gray", alpha=0.45, linewidths=0.8)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(optimizers)))
    for i, name in enumerate(optimizers):
        tr = trajectories[name].numpy()
        ax.plot(tr[:, 0], tr[:, 1], label=name, color=colors[i], linewidth=1.8, alpha=0.9)
        ax.scatter(tr[0, 0], tr[0, 1], color=colors[i], s=40, marker="o", zorder=5)
        ax.scatter(tr[-1, 0], tr[-1, 1], color=colors[i], s=45, marker="x", zorder=5)
    ax.scatter(0, 0, c="black", s=80, marker="*", zorder=6, label="optimum")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Trajectories on $f(x,y)=0.1x^2+y^2$ (same gradient noise per optimizer)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: distance to origin ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(optimizers):
        ax.plot(dists[name].numpy(), label=name, color=colors[i], linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\|\theta - \theta^{*}\|_2$")
    ax.set_title("Distance to optimum (shared noise schedule)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "distance_to_optimum.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: variance of final distance across independent runs ---
    var_rows = []
    for name in optimizers:
        stats = variance_across_runs(
            name,
            cfg,
            init=init,
            lr=lr_for(name),
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            lookahead_k=args.lookahead_k,
            lookahead_alpha=args.lookahead_alpha,
            sigma=args.sigma,
            num_steps=args.steps,
            num_runs=args.var_runs,
            base_seed=1000,
            device=device,
        )
        var_rows.append({"optimizer": name, **stats})

    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(out_dir / "metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = var_df["optimizer"].tolist()
    vals = var_df["var_final_dist"].tolist()
    ax.bar(names, vals, color=colors[: len(names)])
    ax.set_ylabel("Var(final distance to optimum)")
    ax.set_title(f"Across {args.var_runs} independent noise streams (lower = more stable)")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "variance_across_runs.png", dpi=150)
    plt.close(fig)

    # --- Figure 4: Lookahead fast vs slow temporal variance at sync points ---
    la_name: OptimizerName = "lookahead_sgd_momentum"
    la_out = run_quadratic_trajectory(
        la_name,
        cfg,
        init,
        shared_noise,
        lr=lr_for(la_name),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lookahead_k=args.lookahead_k,
        lookahead_alpha=args.lookahead_alpha,
        sigma=args.sigma,
        num_steps=args.steps,
        device=device,
        record_slow_on_sync=True,
    )
    fast_list = la_out["fast_before_sync"]
    slow_list = [s[1].float() for s in la_out["slow_snapshots"]]
    if len(fast_list) == len(slow_list) and fast_list:
        F = torch.stack(fast_list)
        S = torch.stack(slow_list)
        var_f = F.var(dim=0, unbiased=False).sum().item()
        var_s = S.var(dim=0, unbiased=False).sum().item()
        ratio = var_s / var_f if var_f > 1e-12 else float("nan")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["fast @ sync", "slow @ sync"], [var_f, var_s], color=["steelblue", "coral"])
        ax.set_ylabel("Temporal variance (sum over coords)")
        ax.set_title(
            f"Lookahead ({la_name}): temporal variance at sync points (one run)\n"
            f"Var(slow)/Var(fast)={ratio:.4f} — paper α²={args.lookahead_alpha**2:.4f} is a stylized bound, "
            "not this exact statistic"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "lookahead_fast_vs_slow_variance.png", dpi=150)
        plt.close(fig)

        note_path = out_dir / "variance_note.txt"
        note_path.write_text(
            f"Temporal variance at Lookahead sync points (single shared-noise run, {args.steps} steps).\n"
            f"  Sum_i Var_t[f_i]: fast={var_f:.6f}, slow={var_s:.6f}\n"
            f"  Var(slow)/Var(fast) = {ratio:.6f}\n"
            f"  Paper-style scaling reference: alpha^2 = {args.lookahead_alpha**2:.6f} "
            f"(slow weights interpolate toward fast; variance reduction is problem-dependent).\n",
            encoding="utf-8",
        )
    else:
        note_path = out_dir / "variance_note.txt"
        note_path.write_text(
            "Could not align fast/slow snapshots for Lookahead plot (unexpected).\n",
            encoding="utf-8",
        )

    summary = out_dir / "SUMMARY.txt"
    summary.write_text(
        "Quadratic analysis outputs\n"
        "==========================\n"
        f"- trajectories.png: same gradient-noise schedule for every optimizer (fair comparison).\n"
        f"- distance_to_optimum.png: convergence speed / smoothness.\n"
        f"- variance_across_runs.png + metrics.csv: stability over {args.var_runs} noise streams.\n"
        f"- lookahead_fast_vs_slow_variance.png: temporal variance at sync vs alpha^2 reference.\n"
        f"\nHyperparameters: steps={args.steps}, sigma={args.sigma}, k={args.lookahead_k}, "
        f"alpha={args.lookahead_alpha}, lr_sgd={args.lr_sgd}, lr_adam={args.lr_adam}\n",
        encoding="utf-8",
    )

    print(f"Wrote figures and metrics to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
