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

from src.data.dataloaders import get_cifar10_loaders, get_fashion_mnist_loaders, get_mnist_loaders

data_loaders = {
    "cifar10": get_cifar10_loaders,
    "mnist": get_mnist_loaders,
    "fashion_mnist": get_fashion_mnist_loaders,
}
from src.engine.trainer import train_model
from src.models import build_model
from src.optim import Lookahead
from src.utils.seed import set_global_seed


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(
    name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
    momentum: float,
    lookahead_k: int,
    lookahead_alpha: float,
):
    name = name.lower()
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "lookahead_sgd":
        base = SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
        return Lookahead(base_optimizer=base, k=lookahead_k, alpha=lookahead_alpha)
    if name == "lookahead_sgd_momentum":
        base = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        return Lookahead(base_optimizer=base, k=lookahead_k, alpha=lookahead_alpha)
    if name == "lookahead_adam":
        base = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return Lookahead(base_optimizer=base, k=lookahead_k, alpha=lookahead_alpha)
    raise ValueError(
        "Unsupported optimizer. Expected one of: "
        "sgd, sgd_momentum, adam, lookahead_sgd, lookahead_sgd_momentum, lookahead_adam."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline model on CIFAR-10, MNIST, or Fashion-MNIST."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=list(data_loaders),
        help="Dataset to use for training.",
    )
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
    parser.add_argument("--lookahead-k", type=int, default=5)
    parser.add_argument("--lookahead-alpha", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--metrics-dir", type=str, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable download fallback if local dataset files are missing.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    get_loaders = data_loaders[args.dataset]
    train_loader, val_loader, _, class_names = get_loaders(
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
        lookahead_k=args.lookahead_k,
        lookahead_alpha=args.lookahead_alpha,
    )

    run_name = args.run_name or f"{args.dataset}_{args.model}_{args.optimizer}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    dataset_results_root = Path("results") / args.dataset
    log_root = Path(args.log_dir) if args.log_dir is not None else dataset_results_root / "tensorboard"
    checkpoint_root = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir is not None
        else dataset_results_root / "checkpoints"
    )
    metrics_dir = (
        Path(args.metrics_dir)
        if args.metrics_dir is not None
        else dataset_results_root / "logs"
    )
    log_dir = log_root / run_name
    checkpoint_dir = checkpoint_root / run_name
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
