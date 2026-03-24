from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST
from torchvision.transforms import Compose, Lambda, Resize, ToTensor

from src.utils.seed import set_global_seed


def _grayscale_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Replicate grayscale channel to 3 channels for ResNet compatibility."""
    return x.repeat(3, 1, 1)


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

MNIST_CLASSES = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)

FASHION_MNIST_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

# MNIST and Fashion-MNIST are 28x28 grayscale; replicate to 3 channels and resize to 32x32 for ResNet compatibility.
MNIST_TRANSFORM = Compose([
    ToTensor(),
    Lambda(_grayscale_to_rgb),
    Resize((32, 32)),
])


def get_cifar10_loaders(
    data_root: str = "data",
    batch_size: int = 64,
    num_workers: int = 2,
    val_split: float = 0.1,
    seed: int = 42,
    download_if_missing: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[str, ...]]:
    """Create train, validation, and test loaders for CIFAR-10.

    Uses local cifar-10-batches-py under data_root if present; otherwise
    downloads via torchvision when download_if_missing is True.
    """
    root = Path(data_root)
    cifar10_transform = ToTensor()
    train_dataset = CIFAR10(
        root=str(root),
        train=True,
        download=download_if_missing,
        transform=cifar10_transform,
    )
    test_dataset = CIFAR10(
        root=str(root),
        train=False,
        download=download_if_missing,
        transform=cifar10_transform,
    )

    set_global_seed(seed)
    n_train = len(train_dataset)
    indices = torch.randperm(n_train).tolist()
    n_val = int(n_train * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, CIFAR10_CLASSES


def get_mnist_loaders(
    data_root: str = "data",
    batch_size: int = 64,
    num_workers: int = 2,
    val_split: float = 0.1,
    seed: int = 42,
    download_if_missing: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[str, ...]]:
    """Create train, validation, and test loaders for MNIST.

    Uses local MNIST/raw under data_root if present; otherwise downloads via
    torchvision when download_if_missing is True. Transforms grayscale to
    3-channel 32x32 for ResNet compatibility.
    """
    root = Path(data_root)
    train_dataset = MNIST(
        root=str(root),
        train=True,
        download=download_if_missing,
        transform=MNIST_TRANSFORM,
    )
    test_dataset = MNIST(
        root=str(root),
        train=False,
        download=download_if_missing,
        transform=MNIST_TRANSFORM,
    )

    set_global_seed(seed)
    n_train = len(train_dataset)
    indices = torch.randperm(n_train).tolist()
    n_val = int(n_train * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, MNIST_CLASSES


def get_fashion_mnist_loaders(
    data_root: str = "data",
    batch_size: int = 64,
    num_workers: int = 2,
    val_split: float = 0.1,
    seed: int = 42,
    download_if_missing: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[str, ...]]:
    """Create train, validation, and test loaders for Fashion-MNIST.

    Uses local Fashion-MNIST/raw under data_root if present; otherwise downloads
    via torchvision when download_if_missing is True. Transforms grayscale to
    3-channel 32x32 for ResNet compatibility.
    """
    root = Path(data_root)
    train_dataset = FashionMNIST(
        root=str(root),
        train=True,
        download=download_if_missing,
        transform=MNIST_TRANSFORM,
    )
    test_dataset = FashionMNIST(
        root=str(root),
        train=False,
        download=download_if_missing,
        transform=MNIST_TRANSFORM,
    )

    set_global_seed(seed)
    n_train = len(train_dataset)
    indices = torch.randperm(n_train).tolist()
    n_val = int(n_train * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, FASHION_MNIST_CLASSES
