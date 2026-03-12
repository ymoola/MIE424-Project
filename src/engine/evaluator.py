from __future__ import annotations

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
    """Run evaluation loop over a dataloader.

    Computes average loss and accuracy across the provided batches.
    Returns a metrics dictionary compatible with both the training loop
    (expects ``accuracy``) and the standalone evaluation script (expects ``acc``).
    """
    model.eval()

    total_samples = 0
    total_correct = 0
    total_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        batch_size = inputs.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        preds = outputs.max(1, keepdim=True)[1]
        total_correct += preds.eq(labels.view_as(preds)).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0

    return {"loss": avg_loss, "accuracy": avg_acc, "acc": avg_acc}
