import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloaders import get_cifar10_loaders
from src.engine.trainer import train_model
from src.models import build_model
from src.utils.seed import set_global_seed


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float, momentum: float):
    name = name.lower()
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError("Unsupported optimizer. Expected one of: sgd, sgd_momentum, adam.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline model on CIFAR-10.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--optimizer", type=str, default="sgd_momentum")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="results/tensorboard")
    parser.add_argument("--checkpoint-dir", type=str, default="results/checkpoints")
    parser.add_argument("--metrics-dir", type=str, default="results/logs")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable download fallback if local CIFAR-10 files are missing.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names = get_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        download_if_missing=not args.no_download,
    )

    model = build_model(model_name=args.model, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(
        name=args.optimizer,
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    run_name = args.run_name or f"{args.model}_{args.optimizer}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = Path(args.log_dir) / run_name
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        history_df = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            writer=writer,
            checkpoint_dir=checkpoint_dir,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
        )
    finally:
        writer.close()

    metrics_path = metrics_dir / f"{run_name}.csv"
    history_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
