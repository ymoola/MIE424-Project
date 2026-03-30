from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Note that MNIST / Fashion-MNIST are grayscale - these values are the usual single-channel stats repeated for 3 channels
MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)
FASHION_MNIST_MEAN = (0.2860, 0.2860, 0.2860)
FASHION_MNIST_STD = (0.3530, 0.3530, 0.3530)


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


def _build_mnist_train_transform() -> transforms.Compose:
    # convert images to 32x32 to line up with CIFAR and ResNet-style setups
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def _build_mnist_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def _build_fashion_mnist_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ]
    )


def _build_fashion_mnist_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
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


def _resolve_mnist_download_flag(data_root: Path, download_if_missing: bool) -> bool:
    legacy_processed_train = data_root / "MNIST" / "processed" / "training.pt"
    raw_train_images = data_root / "MNIST" / "raw" / "train-images-idx3-ubyte"
    raw_train_labels = data_root / "MNIST" / "raw" / "train-labels-idx1-ubyte"
    raw_test_images = data_root / "MNIST" / "raw" / "t10k-images-idx3-ubyte"
    raw_test_labels = data_root / "MNIST" / "raw" / "t10k-labels-idx1-ubyte"
    if legacy_processed_train.exists() or (
        raw_train_images.exists()
        and raw_train_labels.exists()
        and raw_test_images.exists()
        and raw_test_labels.exists()
    ):
        return False

    if download_if_missing:
        return True

    raise FileNotFoundError(
        "MNIST was not found under "
        f"'{data_root / 'MNIST'}'. "
        "Download the dataset or set download_if_missing=True."
    )


def _resolve_fashion_mnist_download_flag(data_root: Path, download_if_missing: bool) -> bool:
    legacy_processed_train = data_root / "FashionMNIST" / "processed" / "training.pt"
    raw_train_images = data_root / "FashionMNIST" / "raw" / "train-images-idx3-ubyte"
    raw_train_labels = data_root / "FashionMNIST" / "raw" / "train-labels-idx1-ubyte"
    raw_test_images = data_root / "FashionMNIST" / "raw" / "t10k-images-idx3-ubyte"
    raw_test_labels = data_root / "FashionMNIST" / "raw" / "t10k-labels-idx1-ubyte"
    if legacy_processed_train.exists() or (
        raw_train_images.exists()
        and raw_train_labels.exists()
        and raw_test_images.exists()
        and raw_test_labels.exists()
    ):
        return False

    if download_if_missing:
        return True

    raise FileNotFoundError(
        "Fashion-MNIST was not found under "
        f"'{data_root / 'FashionMNIST'}'. "
        "Download the dataset or set download_if_missing=True."
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


def get_mnist_loaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    download_if_missing: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build MNIST train/val/test dataloaders (3-channel 32x32 for ResNet).

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    download = _resolve_mnist_download_flag(data_root, download_if_missing)

    train_transform = _build_mnist_train_transform()
    eval_transform = _build_mnist_eval_transform()

    train_dataset_aug = datasets.MNIST(
        root=str(data_root),
        train=True,
        transform=train_transform,
        download=download,
    )
    train_dataset_eval = datasets.MNIST(
        root=str(data_root),
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_dataset = datasets.MNIST(
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


def get_fashion_mnist_loaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    download_if_missing: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build Fashion-MNIST train/val/test dataloaders (3-channel 32x32 for ResNet).

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    download = _resolve_fashion_mnist_download_flag(data_root, download_if_missing)

    train_transform = _build_fashion_mnist_train_transform()
    eval_transform = _build_fashion_mnist_eval_transform()

    train_dataset_aug = datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        transform=train_transform,
        download=download,
    )
    train_dataset_eval = datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_dataset = datasets.FashionMNIST(
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
