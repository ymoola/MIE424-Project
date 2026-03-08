from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """TODO: Implement one training epoch."""
    raise NotImplementedError("TODO: implement _train_one_epoch in src/engine/trainer.py")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    writer: Optional[SummaryWriter] = None,
    checkpoint_dir: Optional[str | Path] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
) -> pd.DataFrame:
    """TODO: Implement full training loop, logging, and checkpointing."""
    raise NotImplementedError("TODO: implement train_model in src/engine/trainer.py")
