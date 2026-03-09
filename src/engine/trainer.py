from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from evaluator import evaluate_model


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
    
    avg_epoch_loss = total_loss / total_samples
    avg_epoch_acc = total_correct / total_samples

    return {"loss": avg_epoch_loss, "accuracy": avg_epoch_acc}   

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
    """Full training loop, logging, and checkpointing."""

    model.to(device)

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    results = []

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

        # Checkpoint model every 10 epochs
        if (epoch % 10 == 9):
            model_path = Path(checkpoint_dir)/f"checkpoint_epoch_{epoch}.pt"
            torch.save(model.state_dict(), model_path)

    return pd.DataFrame(results)
    
    raise NotImplementedError("TODO: implement train_model in src/engine/trainer.py")
