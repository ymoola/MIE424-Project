import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloaders import get_cifar10_loaders


def _print_batch_info(loader_name: str, loader) -> None:
    images, labels = next(iter(loader))
    print(
        f"{loader_name}: x_shape={tuple(images.shape)} "
        f"y_shape={tuple(labels.shape)} "
        f"label_range=[{labels.min().item()}, {labels.max().item()}]"
    )


def _subset_indices(loader):
    dataset = loader.dataset
    if not hasattr(dataset, "indices"):
        raise RuntimeError("Expected loader.dataset to be a torch.utils.data.Subset.")
    return list(dataset.indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test CIFAR-10 dataloaders.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable torchvision download fallback if local data is missing.",
    )
    args = parser.parse_args()

    download_if_missing = not args.no_download

    train_loader, val_loader, test_loader, class_names = get_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        download_if_missing=download_if_missing,
    )

    _print_batch_info("train", train_loader)
    _print_batch_info("val", val_loader)
    _print_batch_info("test", test_loader)
    print(f"class_count={len(class_names)} classes={class_names}")

    _, val_loader_repeat, _, _ = get_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        download_if_missing=download_if_missing,
    )
    is_deterministic = _subset_indices(val_loader) == _subset_indices(val_loader_repeat)
    print(f"deterministic_val_split={is_deterministic}")


if __name__ == "__main__":
    main()
