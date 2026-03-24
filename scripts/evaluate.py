import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloaders import (
    get_cifar10_loaders,
    get_fashion_mnist_loaders,
    get_mnist_loaders,
)

data_loaders = {
    "cifar10": get_cifar10_loaders,
    "mnist": get_mnist_loaders,
    "fashion_mnist": get_fashion_mnist_loaders,
}
from src.engine.evaluator import evaluate_model
from src.models import build_model


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on CIFAR-10, MNIST, or Fashion-MNIST.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=list(data_loaders), help="Dataset used when training the checkpoint.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable download fallback if local dataset files are missing.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    get_loaders = data_loaders[args.dataset]
    _, val_loader, test_loader, class_names = get_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        download_if_missing=not args.no_download,
    )
    loader = val_loader if args.split == "val" else test_loader

    model = build_model(model_name=args.model, num_classes=len(class_names)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        max_batches=args.max_batches,
    )

    print(
        f"{args.split}_loss={metrics['loss']:.4f} "
        f"{args.split}_acc={metrics['acc']:.4f}"
    )

    if args.output_csv is not None:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "checkpoint": args.checkpoint,
                    "split": args.split,
                    "loss": metrics["loss"],
                    "accuracy": metrics["acc"],
                    "seed": args.seed,
                    "model": args.model,
                }
            ]
        ).to_csv(output_path, index=False)
        print(f"Saved evaluation CSV: {output_path}")


if __name__ == "__main__":
    main()
