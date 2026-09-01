"""
AgriMind AI — Training Pipeline.

Clean reusable training loop with:
- Configurable hyperparameters
- Early stopping
- Best-model checkpointing
- Training history logging
- CPU/CUDA support
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    create_dataloaders,
)
from src.model import build_model, save_checkpoint, unfreeze_backbone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 10
    optimizer: str = "Adam"
    loss_function: str = "CrossEntropyLoss"
    early_stopping_patience: int = 5
    class_weighted_loss: bool = False
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_epoch: int | None = None  # epoch at which to unfreeze
    image_size: int = 224
    num_classes: int = NUM_CLASSES
    classes: list[str] | None = None  # crop-specific class names (index order)
    device: str = "auto"  # "auto", "cpu", or "cuda"
    num_workers: int = 0
    splits_dir: str = "data/splits"
    checkpoint_path: str = "models/best_model.pth"
    metadata_path: str = "models/training_metadata.json"
    history_path: str = "reports/training/training_history.json"
    loss_curve_path: str = "reports/training/loss_curve.png"
    accuracy_curve_path: str = "reports/training/accuracy_curve.png"


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to a torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def load_config(config_path: str = "configs/config.yaml") -> TrainingConfig:
    """Load training config from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config not found at %s, using defaults.", config_path)
        return TrainingConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    training = raw.get("training", {})
    image = raw.get("image", {})
    model_cfg = raw.get("model", {})

    return TrainingConfig(
        seed=raw.get("seed", 42),
        batch_size=training.get("batch_size", 32),
        learning_rate=training.get("learning_rate", 0.001),
        weight_decay=training.get("weight_decay", 1e-4),
        epochs=training.get("epochs", 10),
        optimizer=training.get("optimizer", "Adam"),
        loss_function=training.get("loss_function", "CrossEntropyLoss"),
        early_stopping_patience=training.get("early_stopping_patience", 5),
        class_weighted_loss=training.get("class_weighted_loss", False),
        pretrained=training.get("pretrained", True),
        freeze_backbone=training.get("freeze_backbone", True),
        unfreeze_epoch=training.get("unfreeze_epoch", None),
        image_size=image.get("height", 224),
        num_classes=raw.get("num_classes", NUM_CLASSES),
        classes=raw.get("classes"),
        device=training.get("device", "auto"),
        num_workers=training.get("num_workers", 0),
        splits_dir=raw.get("data", {}).get("splits", "data/splits"),
        checkpoint_path=model_cfg.get("best_model_path", "models/best_model.pth"),
        metadata_path=model_cfg.get("metadata_path", "models/training_metadata.json"),
        history_path=training.get("history_path", "reports/training/training_history.json"),
        loss_curve_path=training.get("loss_curve_path", "reports/training/loss_curve.png"),
        accuracy_curve_path=training.get("accuracy_curve_path", "reports/training/accuracy_curve.png"),
    )


# ---------------------------------------------------------------------------
# Class-weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(
    train_csv: str | Path,
    num_classes: int = NUM_CLASSES,
    class_names: list[str] | None = None,
) -> torch.Tensor:
    """Compute normalised inverse-frequency class weights from a training CSV.

    Formula::

        weight_i = total_samples / (num_classes * count_i)

    The resulting weights are then normalised so their mean equals 1.0.

    Parameters
    ----------
    train_csv : str or Path
        Path to the training manifest CSV (must have ``clean_class`` column).
    num_classes : int
        Number of classes.
    class_names : list of str, optional
        Ordered class names (index order). Defaults to the module-level
        ``IDX_TO_CLASS`` mapping (tomato/potato).

    Returns
    -------
    ``torch.Tensor`` of shape ``(num_classes,)`` with float32 weights ordered
    by class index (``class_names`` order).

    Raises
    ------
    ValueError
        If any class is missing from the training data.
    """
    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    df = pd.read_csv(train_csv)
    counts = Counter(df["clean_class"])

    # Validate all classes present
    missing = [name for name in class_names if name not in counts]
    if missing:
        raise ValueError(
            f"Missing classes in training data: {missing}. "
            f"Cannot compute class weights."
        )

    total = len(df)
    raw_weights = []
    for i in range(num_classes):
        class_name = class_names[i]
        count = counts[class_name]
        raw_weights.append(total / (num_classes * count))

    weights = torch.tensor(raw_weights, dtype=torch.float32)

    # Normalise so mean ≈ 1.0
    weights = weights / weights.mean()

    return weights


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train(
    config: TrainingConfig,
    loaders: dict[str, DataLoader] | None = None,
) -> dict[str, Any]:
    """Run the full training pipeline.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    loaders : dict, optional
        Pre-built DataLoaders. If None, creates from config.splits_dir.

    Returns
    -------
    dict with training history and metadata.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    logger.info("Using device: %s", device)

    # Crop-specific label mapping (defaults to the tomato/potato mapping)
    class_to_idx = (
        {name: i for i, name in enumerate(config.classes)}
        if config.classes
        else None
    )
    class_names = config.classes or [
        IDX_TO_CLASS[i] for i in range(config.num_classes)
    ]

    # Create data loaders
    if loaders is None:
        loaders = create_dataloaders(
            config.splits_dir,
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_workers=config.num_workers,
            class_to_idx=class_to_idx,
        )

    # Build model
    model = build_model(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    )
    model.to(device)

    # Loss and optimizer
    if config.class_weighted_loss:
        train_csv = Path(config.splits_dir) / "train.csv"
        class_weights = compute_class_weights(
            train_csv, config.num_classes, class_names=class_names
        )
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(
            "Class-weighted loss enabled: %s",
            {class_names[i]: f"{class_weights[i]:.4f}" for i in range(config.num_classes)},
        )
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = _create_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # Training history
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    logger.info("=" * 60)
    logger.info("Starting training: %d epochs", config.epochs)
    logger.info("=" * 60)

    start_time = time.time()
    for epoch in range(config.epochs):
        # Unfreeze backbone if configured
        if config.unfreeze_epoch is not None and epoch == config.unfreeze_epoch:
            unfreeze_backbone(model)
            optimizer = _create_optimizer(model, config)

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, loaders["val"], criterion, device)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1, config.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
            checkpoint_dir = Path(config.checkpoint_path).parent
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            metadata = _build_metadata(config, best_val_acc, best_epoch)
            save_checkpoint(model, config.checkpoint_path, metadata=metadata)
            logger.info("New best model saved (val_acc=%.4f)", best_val_acc)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch + 1, config.early_stopping_patience,
                )
                break

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(
        "Training complete in %.1fs — best val_acc=%.4f at epoch %d",
        elapsed, best_val_acc, best_epoch + 1,
    )
    logger.info("=" * 60)

    # Save training metadata
    metadata = _build_metadata(config, best_val_acc, best_epoch, elapsed)
    _save_metadata(metadata, config.metadata_path)

    # Save training history
    _save_history(history, config.history_path)

    # Generate plots
    _plot_training_curves(history, config.loss_curve_path, config.accuracy_curve_path)

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(
            params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            params, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def _build_metadata(
    config: TrainingConfig,
    best_val_acc: float,
    best_epoch: int,
    elapsed: float | None = None,
) -> dict[str, Any]:
    """Build metadata dict for checkpoint/report."""
    meta = {
        "architecture": "mobilenet_v3_small",
        "num_classes": config.num_classes,
        "class_mapping": (
            {i: name for i, name in enumerate(config.classes)}
            if config.classes
            else {v: k for k, v in CLASS_TO_IDX.items()}
        ),
        "input_size": config.image_size,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
        "optimizer": config.optimizer,
        "pretrained": config.pretrained,
        "freeze_backbone": config.freeze_backbone,
        "class_weighted_loss": config.class_weighted_loss,
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch + 1,
    }
    if elapsed is not None:
        meta["elapsed_seconds"] = elapsed
    return meta


def _save_metadata(metadata: dict[str, Any], path: str) -> None:
    """Save training metadata as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Training metadata saved: %s", p)


def _save_history(history: dict[str, list[float]], path: str) -> None:
    """Save training history as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("Training history saved: %s", p)


def _plot_training_curves(
    history: dict[str, list[float]],
    loss_path: str,
    acc_path: str,
) -> None:
    """Generate and save training curve plots."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    Path(loss_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    ax.plot(epochs, history["val_acc"], label="Val Accuracy", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    Path(acc_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)

    logger.info("Training curves saved.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python -m src.train``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="AgriMind AI — Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], help="Override device"
    )
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Use randomly initialized weights"
    )
    parser.add_argument(
        "--unfreeze-epoch", type=int, help="Epoch to unfreeze backbone"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.device is not None:
        config.device = args.device
    if args.no_pretrained:
        config.pretrained = False
    if args.unfreeze_epoch is not None:
        config.unfreeze_epoch = args.unfreeze_epoch

    result = train(config)
    logger.info("Best validation accuracy: %.4f", result["best_val_acc"])


if __name__ == "__main__":
    main()
