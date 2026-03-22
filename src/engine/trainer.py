from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.engine.evaluator import evaluate_model


def _compute_grad_norm(model: nn.Module) -> float:
    total_squared_norm = 0.0
    found_gradient = False

    for param in model.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().norm(2).item()
        total_squared_norm += grad_norm * grad_norm
        found_gradient = True

    if not found_gradient:
        return 0.0

    return total_squared_norm ** 0.5


def _get_learning_rate(optimizer: Optimizer) -> float:
    if not optimizer.param_groups:
        raise RuntimeError("Optimizer has no parameter groups.")
    return float(optimizer.param_groups[0]["lr"])


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
    total_loss = 0.0
    total_grad_norm = 0.0
    batch_count = 0
    lookahead_distance_total = 0.0
    lookahead_sync_count = 0

    for i, (inputs, labels) in enumerate(dataloader, 0):

        if max_batches is not None and i >= max_batches:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        total_grad_norm += _compute_grad_norm(model)
        optimizer.step()
        batch_count += 1

        if getattr(optimizer, "sync_happened", False):
            sync_distance = getattr(optimizer, "last_sync_distance", None)
            if sync_distance is not None:
                lookahead_distance_total += float(sync_distance)
                lookahead_sync_count += 1

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
    avg_grad_norm = total_grad_norm / batch_count if batch_count > 0 else 0.0

    metrics = {
        "loss": avg_epoch_loss,
        "accuracy": avg_epoch_acc,
        "grad_norm": avg_grad_norm,
    }
    if lookahead_sync_count > 0:
        metrics["fast_slow_distance"] = lookahead_distance_total / lookahead_sync_count

    return metrics


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
        epoch_start = perf_counter()
        train_values = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_train_batches,
        )

        avg_train_loss = train_values["loss"]
        avg_train_acc = train_values["accuracy"]
        avg_grad_norm = train_values["grad_norm"]

        val_values = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            max_eval_batches,
        )

        avg_val_loss = val_values["loss"]
        avg_val_acc = val_values["accuracy"]
        learning_rate = _get_learning_rate(optimizer)
        generalization_gap_acc = avg_train_acc - avg_val_acc
        generalization_gap_loss = avg_val_loss - avg_train_loss
        epoch_time_sec = perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_acc,
            "learning_rate": learning_rate,
            "generalization_gap_acc": generalization_gap_acc,
            "generalization_gap_loss": generalization_gap_loss,
            "epoch_time_sec": epoch_time_sec,
            "grad_norm": avg_grad_norm,
        }
        if "fast_slow_distance" in train_values:
            row["fast_slow_distance"] = train_values["fast_slow_distance"]
        results.append(row)

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Accuracy/train", avg_train_acc, epoch)
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/validation", avg_val_acc, epoch)
            writer.add_scalar("Optimizer/learning_rate", learning_rate, epoch)
            writer.add_scalar("Optimizer/grad_norm", avg_grad_norm, epoch)
            writer.add_scalar("Diagnostics/generalization_gap_acc", generalization_gap_acc, epoch)
            writer.add_scalar("Diagnostics/generalization_gap_loss", generalization_gap_loss, epoch)
            writer.add_scalar("Diagnostics/epoch_time_sec", epoch_time_sec, epoch)
            if "fast_slow_distance" in train_values:
                writer.add_scalar(
                    "Lookahead/fast_slow_distance",
                    train_values["fast_slow_distance"],
                    epoch,
                )

        if checkpoint_root is not None:
            # Save latest every epoch so short runs always produce a checkpoint.
            latest_path = checkpoint_root / "latest.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": row,
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
                        "metrics": row,
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
                        "metrics": row,
                    },
                    periodic_path,
                )

    return pd.DataFrame(results)
