import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate.py"
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"

DATASET_CHOICES = ("cifar10", "mnist", "fashion_mnist")


def _dataset_hparams(dataset: str) -> dict[str, object]:
    """Pilot grids and regularization tuned per dataset (10-class ResNet on 32x32)."""
    if dataset == "mnist":
        return {
            "weight_decay": 1e-4,
            "pilot_epochs": 10,
            "sgd_lr_grid": [0.1, 0.2, 0.3],
            "adam_lr_grid": [1e-3, 3e-3, 1e-2],
            "lookahead_epochs": 25,
            "core_epochs": 100,
            "final_epochs": 100,
        }
    if dataset == "fashion_mnist":
        return {
            "weight_decay": 5e-4,
            "pilot_epochs": 12,
            "sgd_lr_grid": [0.05, 0.1, 0.2],
            "adam_lr_grid": [5e-4, 1e-3, 3e-3],
            "lookahead_epochs": 30,
            "core_epochs": 100,
            "final_epochs": 100,
        }
    return {
        "weight_decay": 5e-4,
        "pilot_epochs": 10,
        "sgd_lr_grid": [0.03, 0.05, 0.1],
        "adam_lr_grid": [3e-4, 1e-3, 3e-3],
        "lookahead_epochs": 30,
        "core_epochs": 100,
        "final_epochs": 100,
    }


def _dataset_experiments_root(dataset: str) -> Path:
    """Per-dataset folder so MNIST / CIFAR / Fashion runs do not overwrite each other's CSVs."""
    return EXPERIMENTS_ROOT / dataset


def _manifest_path(dataset: str) -> Path:
    return _dataset_experiments_root(dataset) / "manifest.csv"


def _ensure_experiments_root(dataset: str) -> None:
    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    _dataset_experiments_root(dataset).mkdir(parents=True, exist_ok=True)


def _suite_experiments_dir(suite_name: str, dataset: str) -> Path:
    return _dataset_experiments_root(dataset) / suite_name


def _suite_logs_dir(suite_name: str) -> Path:
    return RESULTS_ROOT / "logs" / suite_name


def _suite_checkpoints_dir(suite_name: str) -> Path:
    return RESULTS_ROOT / "checkpoints" / suite_name


def _suite_tensorboard_dir(suite_name: str) -> Path:
    return RESULTS_ROOT / "tensorboard" / suite_name


def _relative_to_repo(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _format_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _make_run_name(config: dict[str, object], dataset: str) -> str:
    name_parts = [
        str(config["suite"]),
        str(config["optimizer"]),
        f"lr{_format_float(float(config['lr']))}",
        f"s{config['seed']}",
    ]
    if "lookahead" in str(config["optimizer"]):
        name_parts.append(f"k{config['lookahead_k']}")
        name_parts.append(f"a{_format_float(float(config['lookahead_alpha']))}")
    return "__".join(name_parts) + f"__{dataset}"


def _metrics_path(suite_name: str, run_name: str) -> Path:
    return _suite_logs_dir(suite_name) / f"{run_name}.csv"


def _checkpoint_dir(suite_name: str, run_name: str) -> Path:
    return _suite_checkpoints_dir(suite_name) / run_name


def _run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def _read_manifest(manifest_path: Path) -> pd.DataFrame:
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    return pd.DataFrame(
        columns=[
            "suite",
            "run_name",
            "optimizer",
            "lr",
            "momentum",
            "weight_decay",
            "lookahead_k",
            "lookahead_alpha",
            "seed",
            "epochs",
            "metrics_csv",
            "checkpoint_dir",
        ]
    )


def _write_manifest(rows: list[dict[str, object]], manifest_path: Path) -> None:
    manifest_df = _read_manifest(manifest_path)
    new_rows_df = pd.DataFrame(rows)
    merged = pd.concat([manifest_df, new_rows_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["suite", "run_name"], keep="last")
    merged.to_csv(manifest_path, index=False)


def _load_metrics(metrics_csv: Path) -> pd.DataFrame:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")
    return pd.read_csv(metrics_csv)


def _summarize_run_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    best_idx = int(metrics_df["val_accuracy"].idxmax())
    best_row = metrics_df.loc[best_idx]
    final_row = metrics_df.iloc[-1]
    return {
        "best_val_accuracy": float(best_row["val_accuracy"]),
        "best_val_epoch": int(best_row["epoch"]),
        "final_val_accuracy": float(final_row["val_accuracy"]),
        "final_val_loss": float(final_row["val_loss"]),
        "final_train_loss": float(final_row["train_loss"]),
    }


def _train_run(
    config: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    suite_name = str(config["suite"])
    run_name = _make_run_name(config, args.dataset)
    metrics_csv = _metrics_path(suite_name, run_name)
    checkpoint_dir = _checkpoint_dir(suite_name, run_name)

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--val-split",
        str(args.val_split),
        "--seed",
        str(config["seed"]),
        "--epochs",
        str(config["epochs"]),
        "--model",
        args.model,
        "--optimizer",
        str(config["optimizer"]),
        "--lr",
        str(config["lr"]),
        "--momentum",
        str(config["momentum"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--lookahead-k",
        str(config["lookahead_k"]),
        "--lookahead-alpha",
        str(config["lookahead_alpha"]),
        "--device",
        args.device,
        "--run-name",
        run_name,
        "--log-dir",
        str(_suite_tensorboard_dir(suite_name)),
        "--checkpoint-dir",
        str(_suite_checkpoints_dir(suite_name)),
        "--metrics-dir",
        str(_suite_logs_dir(suite_name)),
    ]
    if args.no_download:
        command.append("--no-download")
    if args.max_train_batches is not None:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(args.max_eval_batches)])

    _run_command(command)

    row = {
        "suite": suite_name,
        "run_name": run_name,
        "optimizer": config["optimizer"],
        "lr": config["lr"],
        "momentum": config["momentum"],
        "weight_decay": config["weight_decay"],
        "lookahead_k": config["lookahead_k"],
        "lookahead_alpha": config["lookahead_alpha"],
        "seed": config["seed"],
        "epochs": config["epochs"],
        "metrics_csv": _relative_to_repo(metrics_csv),
        "checkpoint_dir": _relative_to_repo(checkpoint_dir),
    }
    _write_manifest([row], _manifest_path(args.dataset))
    return row


def _load_required_csv(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Run the prerequisite suite first."
        )
    return pd.read_csv(path)


def _select_pilot_lrs(pilot_runs: pd.DataFrame, dataset: str) -> pd.DataFrame:
    selections = []
    family_map = {
        "sgd_family": ["sgd_momentum", "lookahead_sgd_momentum"],
        "adam_family": ["adam", "lookahead_adam"],
    }

    for family, optimizers in family_map.items():
        family_runs = pilot_runs[pilot_runs["optimizer"].isin(optimizers)].copy()
        if family_runs.empty:
            continue
        grouped = (
            family_runs.groupby("lr", as_index=False)["final_val_accuracy"]
            .mean()
            .sort_values(["final_val_accuracy", "lr"], ascending=[False, True])
        )
        best = grouped.iloc[0]
        selections.append(
            {
                "family": family,
                "selected_lr": float(best["lr"]),
                "selection_score": float(best["final_val_accuracy"]),
            }
        )

    selected_df = pd.DataFrame(selections)
    out_root = _dataset_experiments_root(dataset)
    selected_df.to_csv(out_root / "pilot_lr_selected.csv", index=False)
    suite_dir = _suite_experiments_dir("pilot_lr", dataset)
    suite_dir.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(suite_dir / "selected.csv", index=False)
    return selected_df


def _pilot_lr_configs(dataset: str) -> list[dict[str, object]]:
    hp = _dataset_hparams(dataset)
    wd = float(hp["weight_decay"])
    pilot_epochs = int(hp["pilot_epochs"])
    sgd_lr_grid = [float(x) for x in hp["sgd_lr_grid"]]  # type: ignore[arg-type]
    adam_lr_grid = [float(x) for x in hp["adam_lr_grid"]]  # type: ignore[arg-type]

    configs = []
    for optimizer in ["sgd_momentum", "lookahead_sgd_momentum"]:
        for lr in sgd_lr_grid:
            configs.append(
                {
                    "suite": "pilot_lr",
                    "optimizer": optimizer,
                    "lr": lr,
                    "momentum": 0.9,
                    "weight_decay": wd,
                    "lookahead_k": 5,
                    "lookahead_alpha": 0.5,
                    "seed": 42,
                    "epochs": pilot_epochs,
                }
            )
    for optimizer in ["adam", "lookahead_adam"]:
        for lr in adam_lr_grid:
            configs.append(
                {
                    "suite": "pilot_lr",
                    "optimizer": optimizer,
                    "lr": lr,
                    "momentum": 0.9,
                    "weight_decay": wd,
                    "lookahead_k": 5,
                    "lookahead_alpha": 0.5,
                    "seed": 42,
                    "epochs": pilot_epochs,
                }
            )
    return configs


def _load_selected_lrs(dataset: str) -> dict[str, float]:
    selected_df = _load_required_csv(
        _dataset_experiments_root(dataset) / "pilot_lr_selected.csv",
        "pilot LR selections",
    )
    return {
        row["family"]: float(row["selected_lr"])
        for _, row in selected_df.iterrows()
    }


def _core_comparison_configs(selected_lrs: dict[str, float], dataset: str) -> list[dict[str, object]]:
    hp = _dataset_hparams(dataset)
    wd = float(hp["weight_decay"])
    epochs = int(hp["core_epochs"])
    sgd_lr = selected_lrs["sgd_family"]
    adam_lr = selected_lrs["adam_family"]
    return [
        {
            "suite": "core_comparison",
            "optimizer": "sgd",
            "lr": sgd_lr,
            "momentum": 0.0,
            "weight_decay": wd,
            "lookahead_k": 5,
            "lookahead_alpha": 0.5,
            "seed": 42,
            "epochs": epochs,
        },
        {
            "suite": "core_comparison",
            "optimizer": "sgd_momentum",
            "lr": sgd_lr,
            "momentum": 0.9,
            "weight_decay": wd,
            "lookahead_k": 5,
            "lookahead_alpha": 0.5,
            "seed": 42,
            "epochs": epochs,
        },
        {
            "suite": "core_comparison",
            "optimizer": "adam",
            "lr": adam_lr,
            "momentum": 0.9,
            "weight_decay": wd,
            "lookahead_k": 5,
            "lookahead_alpha": 0.5,
            "seed": 42,
            "epochs": epochs,
        },
        {
            "suite": "core_comparison",
            "optimizer": "lookahead_sgd_momentum",
            "lr": sgd_lr,
            "momentum": 0.9,
            "weight_decay": wd,
            "lookahead_k": 5,
            "lookahead_alpha": 0.5,
            "seed": 42,
            "epochs": epochs,
        },
        {
            "suite": "core_comparison",
            "optimizer": "lookahead_adam",
            "lr": adam_lr,
            "momentum": 0.9,
            "weight_decay": wd,
            "lookahead_k": 5,
            "lookahead_alpha": 0.5,
            "seed": 42,
            "epochs": epochs,
        },
    ]


def _lookahead_sensitivity_configs(selected_lrs: dict[str, float], dataset: str) -> list[dict[str, object]]:
    hp = _dataset_hparams(dataset)
    wd = float(hp["weight_decay"])
    epochs = int(hp["lookahead_epochs"])
    configs = []
    sweep = [
        (5, 0.2),
        (5, 0.5),
        (5, 0.8),
        (10, 0.5),
        (20, 0.5),
    ]
    for optimizer, lr in [
        ("lookahead_sgd_momentum", selected_lrs["sgd_family"]),
        ("lookahead_adam", selected_lrs["adam_family"]),
    ]:
        for k_value, alpha_value in sweep:
            configs.append(
                {
                    "suite": "lookahead_sensitivity",
                    "optimizer": optimizer,
                    "lr": lr,
                    "momentum": 0.9,
                    "weight_decay": wd,
                    "lookahead_k": k_value,
                    "lookahead_alpha": alpha_value,
                    "seed": 42,
                    "epochs": epochs,
                }
            )
    return configs


def _select_lookahead_configs(sensitivity_runs: pd.DataFrame, dataset: str) -> pd.DataFrame:
    selections = []
    default_pref = {
        "lookahead_sgd_momentum": (5, 0.5),
        "lookahead_adam": (5, 0.5),
    }

    for optimizer in ["lookahead_sgd_momentum", "lookahead_adam"]:
        optimizer_runs = sensitivity_runs[sensitivity_runs["optimizer"] == optimizer].copy()
        if optimizer_runs.empty:
            continue
        preferred_k, preferred_alpha = default_pref[optimizer]
        optimizer_runs["preferred_default"] = (
            (optimizer_runs["lookahead_k"] == preferred_k)
            & (optimizer_runs["lookahead_alpha"] == preferred_alpha)
        )
        optimizer_runs = optimizer_runs.sort_values(
            ["final_val_accuracy", "preferred_default"],
            ascending=[False, False],
        )
        best = optimizer_runs.iloc[0]
        selections.append(
            {
                "optimizer": optimizer,
                "lr": float(best["lr"]),
                "lookahead_k": int(best["lookahead_k"]),
                "lookahead_alpha": float(best["lookahead_alpha"]),
                "final_val_accuracy": float(best["final_val_accuracy"]),
            }
        )

    selected_df = pd.DataFrame(selections)
    out_root = _dataset_experiments_root(dataset)
    selected_df.to_csv(out_root / "lookahead_sensitivity_selected.csv", index=False)
    suite_dir = _suite_experiments_dir("lookahead_sensitivity", dataset)
    suite_dir.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(suite_dir / "selected.csv", index=False)
    return selected_df


def _load_selected_lookahead_configs(dataset: str) -> dict[str, dict[str, float]]:
    selected_df = _load_required_csv(
        _dataset_experiments_root(dataset) / "lookahead_sensitivity_selected.csv",
        "lookahead sensitivity selections",
    )
    selected = {}
    for _, row in selected_df.iterrows():
        selected[str(row["optimizer"])] = {
            "lr": float(row["lr"]),
            "lookahead_k": int(row["lookahead_k"]),
            "lookahead_alpha": float(row["lookahead_alpha"]),
        }
    return selected


def _final_repeats_configs(
    selected_lrs: dict[str, float],
    selected_lookahead: dict[str, dict[str, float]],
    dataset: str,
) -> list[dict[str, object]]:
    hp = _dataset_hparams(dataset)
    wd = float(hp["weight_decay"])
    epochs = int(hp["final_epochs"])
    configs = []
    for seed in [42, 52, 62]:
        configs.extend(
            [
                {
                    "suite": "final_repeats",
                    "optimizer": "sgd_momentum",
                    "lr": selected_lrs["sgd_family"],
                    "momentum": 0.9,
                    "weight_decay": wd,
                    "lookahead_k": 5,
                    "lookahead_alpha": 0.5,
                    "seed": seed,
                    "epochs": epochs,
                },
                {
                    "suite": "final_repeats",
                    "optimizer": "adam",
                    "lr": selected_lrs["adam_family"],
                    "momentum": 0.9,
                    "weight_decay": wd,
                    "lookahead_k": 5,
                    "lookahead_alpha": 0.5,
                    "seed": seed,
                    "epochs": epochs,
                },
                {
                    "suite": "final_repeats",
                    "optimizer": "lookahead_sgd_momentum",
                    "lr": selected_lookahead["lookahead_sgd_momentum"]["lr"],
                    "momentum": 0.9,
                    "weight_decay": wd,
                    "lookahead_k": selected_lookahead["lookahead_sgd_momentum"]["lookahead_k"],
                    "lookahead_alpha": selected_lookahead["lookahead_sgd_momentum"]["lookahead_alpha"],
                    "seed": seed,
                    "epochs": epochs,
                },
                {
                    "suite": "final_repeats",
                    "optimizer": "lookahead_adam",
                    "lr": selected_lookahead["lookahead_adam"]["lr"],
                    "momentum": 0.9,
                    "weight_decay": wd,
                    "lookahead_k": selected_lookahead["lookahead_adam"]["lookahead_k"],
                    "lookahead_alpha": selected_lookahead["lookahead_adam"]["lookahead_alpha"],
                    "seed": seed,
                    "epochs": epochs,
                },
            ]
        )
    return configs


def _apply_limit(configs: list[dict[str, object]], limit_runs: int | None) -> list[dict[str, object]]:
    if limit_runs is None:
        return configs
    return configs[:limit_runs]


def _run_training_suite(
    suite_name: str,
    configs: list[dict[str, object]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for config in _apply_limit(configs, args.limit_runs):
        run_row = _train_run(config, args)
        metrics_df = _load_metrics(PROJECT_ROOT / run_row["metrics_csv"])
        run_summary = _summarize_run_metrics(metrics_df)
        rows.append({**run_row, **run_summary})

    suite_df = pd.DataFrame(rows)
    out_root = _dataset_experiments_root(args.dataset)
    suite_df.to_csv(out_root / f"{suite_name}_runs.csv", index=False)
    suite_dir = _suite_experiments_dir(suite_name, args.dataset)
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_df.to_csv(suite_dir / "runs.csv", index=False)
    return suite_df


def _save_group_summary(
    runs_df: pd.DataFrame,
    group_column: str,
    value_columns: list[str],
    output_path: Path,
) -> None:
    summary = (
        runs_df.groupby(group_column)[value_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        group_column if col == (group_column, "") else "_".join(filter(None, col)).strip("_")
        for col in summary.columns.to_flat_index()
    ]
    summary.to_csv(output_path, index=False)


def _run_pilot_lr(args: argparse.Namespace) -> None:
    pilot_runs = _run_training_suite("pilot_lr", _pilot_lr_configs(args.dataset), args)
    _select_pilot_lrs(pilot_runs, args.dataset)


def _run_core_comparison(args: argparse.Namespace) -> None:
    selected_lrs = _load_selected_lrs(args.dataset)
    core_runs = _run_training_suite(
        "core_comparison",
        _core_comparison_configs(selected_lrs, args.dataset),
        args,
    )
    out_root = _dataset_experiments_root(args.dataset)
    core_runs.to_csv(out_root / "core_comparison_summary.csv", index=False)
    suite_dir = _suite_experiments_dir("core_comparison", args.dataset)
    suite_dir.mkdir(parents=True, exist_ok=True)
    core_runs.to_csv(suite_dir / "summary.csv", index=False)


def _run_lookahead_sensitivity(args: argparse.Namespace) -> None:
    selected_lrs = _load_selected_lrs(args.dataset)
    sensitivity_runs = _run_training_suite(
        "lookahead_sensitivity",
        _lookahead_sensitivity_configs(selected_lrs, args.dataset),
        args,
    )
    _select_lookahead_configs(sensitivity_runs, args.dataset)
    suite_dir = _suite_experiments_dir("lookahead_sensitivity", args.dataset)
    suite_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_runs.to_csv(suite_dir / "summary.csv", index=False)


def _run_final_repeats(args: argparse.Namespace) -> None:
    selected_lrs = _load_selected_lrs(args.dataset)
    selected_lookahead = _load_selected_lookahead_configs(args.dataset)
    final_runs = _run_training_suite(
        "final_repeats",
        _final_repeats_configs(selected_lrs, selected_lookahead, args.dataset),
        args,
    )
    final_runs = final_runs[
        [
            "optimizer",
            "seed",
            "run_name",
            "best_val_accuracy",
            "best_val_epoch",
            "final_val_accuracy",
            "final_val_loss",
        ]
    ]
    out_root = _dataset_experiments_root(args.dataset)
    final_runs.to_csv(out_root / "final_repeats_runs.csv", index=False)
    suite_dir = _suite_experiments_dir("final_repeats", args.dataset)
    suite_dir.mkdir(parents=True, exist_ok=True)
    final_runs.to_csv(suite_dir / "runs.csv", index=False)
    _save_group_summary(
        final_runs,
        "optimizer",
        ["best_val_accuracy", "final_val_accuracy"],
        out_root / "final_repeats_summary.csv",
    )
    summary_path = suite_dir / "summary.csv"
    summary_df = pd.read_csv(out_root / "final_repeats_summary.csv")
    summary_df.to_csv(summary_path, index=False)


def _run_final_test(args: argparse.Namespace) -> None:
    out_root = _dataset_experiments_root(args.dataset)
    final_repeats = _load_required_csv(
        out_root / "final_repeats_runs.csv",
        "final repeats summary",
    )
    eval_rows = []
    eval_output_dir = _suite_experiments_dir("final_test", args.dataset) / "evals"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in final_repeats.head(args.limit_runs).iterrows() if args.limit_runs is not None else final_repeats.iterrows():
        best_checkpoint = _suite_checkpoints_dir("final_repeats") / str(row["run_name"]) / "best.pt"
        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Expected best checkpoint not found: {best_checkpoint}")

        output_csv = eval_output_dir / f"{row['run_name']}_test.csv"
        command = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--checkpoint",
            str(best_checkpoint),
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--val-split",
            str(args.val_split),
            "--seed",
            str(int(row["seed"])),
            "--model",
            args.model,
            "--device",
            args.device,
            "--split",
            "test",
            "--output-csv",
            str(output_csv),
        ]
        if args.no_download:
            command.append("--no-download")
        if args.max_eval_batches is not None:
            command.extend(["--max-batches", str(args.max_eval_batches)])

        _run_command(command)
        eval_df = pd.read_csv(output_csv)
        eval_rows.append(
            {
                "optimizer": row["optimizer"],
                "seed": int(row["seed"]),
                "run_name": row["run_name"],
                "checkpoint": str(best_checkpoint.relative_to(PROJECT_ROOT)),
                "loss": float(eval_df.iloc[0]["loss"]),
                "accuracy": float(eval_df.iloc[0]["accuracy"]),
            }
        )

    final_test_df = pd.DataFrame(eval_rows)
    final_test_df.to_csv(out_root / "final_test_runs.csv", index=False)
    suite_dir = _suite_experiments_dir("final_test", args.dataset)
    suite_dir.mkdir(parents=True, exist_ok=True)
    final_test_df.to_csv(suite_dir / "runs.csv", index=False)
    _save_group_summary(
        final_test_df,
        "optimizer",
        ["accuracy", "loss"],
        out_root / "final_test_summary.csv",
    )
    summary_df = pd.read_csv(out_root / "final_test_summary.csv")
    summary_df.to_csv(suite_dir / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimizer experiment suites (CIFAR-10, MNIST, or Fashion-MNIST).")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=[
            "all",
            "pilot_lr",
            "core_comparison",
            "lookahead_sensitivity",
            "final_repeats",
            "final_test",
        ],
    )
    parser.add_argument("--dataset", type=str, default="cifar10", choices=DATASET_CHOICES)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow torchvision download if local dataset files are missing.",
    )
    args = parser.parse_args()
    args.no_download = not args.allow_download

    _ensure_experiments_root(args.dataset)

    if args.suite in ("all", "pilot_lr"):
        _run_pilot_lr(args)
    if args.suite in ("all", "core_comparison"):
        _run_core_comparison(args)
    if args.suite in ("all", "lookahead_sensitivity"):
        _run_lookahead_sensitivity(args)
    if args.suite in ("all", "final_repeats"):
        _run_final_repeats(args)
    if args.suite in ("all", "final_test"):
        _run_final_test(args)


if __name__ == "__main__":
    main()
