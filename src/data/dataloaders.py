from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _resolve_download_flag(data_root: Path, download_if_missing: bool) -> bool:
    local_cifar_folder = data_root / "cifar-10-batches-py"
    if local_cifar_folder.exists():
        return False

    if download_if_missing:
        return True

    raise FileNotFoundError(
        "CIFAR-10 was not found at "
        f"'{local_cifar_folder}'. "
        "Place 'cifar-10-batches-py' there or enable download_if_missing."
    )


def _split_indices(num_items: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1 (exclusive). Got {val_split}.")

    num_val = int(num_items * val_split)
    if num_val <= 0 or num_val >= num_items:
        raise ValueError(
            f"val_split={val_split} produced invalid split sizes for num_items={num_items}."
        )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_items, generator=generator).tolist()
    val_indices = permutation[:num_val]
    train_indices = permutation[num_val:]
    return train_indices, val_indices


def get_cifar10_loaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    download_if_missing: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build CIFAR-10 train/val/test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    download = _resolve_download_flag(data_root=data_root, download_if_missing=download_if_missing)

    train_transform = _build_train_transform()
    eval_transform = _build_eval_transform()

    train_dataset_aug = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        transform=train_transform,
        download=download,
    )
    train_dataset_eval = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        transform=eval_transform,
        download=False,
    )

    train_indices, val_indices = _split_indices(
        num_items=len(train_dataset_aug),
        val_split=val_split,
        seed=seed,
    )

    train_subset = Subset(train_dataset_aug, train_indices)
    val_subset = Subset(train_dataset_eval, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_names = list(train_dataset_aug.classes)
    return train_loader, val_loader, test_loader, class_names
