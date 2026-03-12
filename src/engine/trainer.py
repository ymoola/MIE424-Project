from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.engine.evaluator import evaluate_model


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """One training epoch."""

    model.train()

    total_samples = 0
    total_correct = 0
    total_loss = 0

    for i, (inputs, labels) in enumerate(dataloader, 0):

        if max_batches is not None and i >= max_batches:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        # Saving the current training information
        batch_size = inputs.shape[0]
        total_samples += batch_size

        total_loss += loss.item() * batch_size

        pred = out.max(1, keepdim=True)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum().item()
    
    if total_samples == 0:
        raise RuntimeError("No training samples were processed. Check dataloader or max_batches.")

    avg_epoch_loss = total_loss / total_samples
    avg_epoch_acc = total_correct / total_samples

    return {"loss": avg_epoch_loss, "accuracy": avg_epoch_acc}   

    


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
    """Full training loop, logging, and checkpointing."""

    model.to(device)

    checkpoint_root: Optional[Path] = None
    if checkpoint_dir is not None:
        checkpoint_root = Path(checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    results = []
    best_val_acc = float("-inf")

    for epoch in range(epochs):
        train_values = _train_one_epoch(model, 
                                  train_loader, 
                                  criterion, 
                                  optimizer, 
                                  device, 
                                  max_train_batches)
        
        avg_train_loss = train_values["loss"]
        avg_train_acc = train_values["accuracy"]

        val_values = evaluate_model(model,
                                    val_loader,
                                    criterion,
                                    device,
                                    max_eval_batches)
        
        avg_val_loss = val_values["loss"]
        avg_val_acc = val_values["accuracy"]

        results.append({
            "epoch": epoch,
            "train loss": avg_train_loss,
            "train accuracy": avg_train_acc,
            "val loss": avg_val_loss,
            "val accuracy": avg_val_acc
        })

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Accuracy/train", avg_train_acc, epoch)
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/validation", avg_val_acc, epoch)

        if checkpoint_root is not None:
            # Save latest every epoch so short runs always produce a checkpoint.
            latest_path = checkpoint_root / "latest.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": results[-1],
                },
                latest_path,
            )

            # Save best by validation accuracy for easy model selection.
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_path = checkpoint_root / "best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "metrics": results[-1],
                    },
                    best_path,
                )

            # Keep periodic snapshots every 10 epochs for long runs.
            if epoch % 10 == 9:
                periodic_path = checkpoint_root / f"checkpoint_epoch_{epoch}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "metrics": results[-1],
                    },
                    periodic_path,
                )

    return pd.DataFrame(results)
    
    
