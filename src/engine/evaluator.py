from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """TODO: Implement validation/test evaluation loop."""
    raise NotImplementedError("TODO: implement evaluate_model in src/engine/evaluator.py")
