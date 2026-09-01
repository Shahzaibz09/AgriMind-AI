"""
AgriMind AI — Model Evaluation.

Provides reusable evaluation functions for accuracy, precision, recall,
F1, per-class metrics, confusion matrix, and prediction probabilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.dataset import IDX_TO_CLASS, NUM_CLASSES
from src.model import load_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Run inference on a DataLoader and collect predictions.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    loader : DataLoader
        Evaluation data loader.
    device : torch.device or str
        Device for inference.

    Returns
    -------
    dict with:
        - ``y_true``: ground-truth labels (int array)
        - ``y_pred``: predicted labels (int array)
        - ``probabilities``: softmax probabilities (N x num_classes)
        - ``image_paths``: list of image paths
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    model.eval()

    all_true: list[int] = []
    all_pred: list[int] = []
    all_probs: list[np.ndarray] = []
    all_paths: list[str] = []

    for images, labels, paths in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_true.extend(labels.numpy().tolist())
        all_pred.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_paths.extend(paths)

    return {
        "y_true": np.array(all_true),
        "y_pred": np.array(all_pred),
        "probabilities": np.array(all_probs),
        "image_paths": all_paths,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : array of int
        Ground-truth labels.
    y_pred : array of int
        Predicted labels.
    class_names : list of str, optional
        Class names for per-class breakdown.

    Returns
    -------
    dict with overall and per-class metrics.
    """
    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        y_bin_true = (y_true == i).astype(int)
        y_bin_pred = (y_pred == i).astype(int)
        per_class[name] = {
            "precision": float(precision_score(y_bin_true, y_bin_pred, zero_division=0)),
            "recall": float(recall_score(y_bin_true, y_bin_pred, zero_division=0)),
            "f1": float(f1_score(y_bin_true, y_bin_pred, zero_division=0)),
            "support": int(np.sum(y_true == i)),
        }

    report_str = classification_report(
        y_true, y_pred, target_names=class_names,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    return {
        "overall": overall,
        "per_class": per_class,
        "classification_report": report_str,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> np.ndarray:
    """Compute confusion matrix."""
    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    return confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Plot and save confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", output_path)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run full evaluation pipeline.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    loader : DataLoader
        Evaluation DataLoader (val or test).
    device : str or torch.device
        Device for inference.
    class_names : list of str, optional
        Class names.

    Returns
    -------
    dict with predictions and metrics.
    """
    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

    preds = predict(model, loader, device=device)
    metrics = compute_metrics(preds["y_true"], preds["y_pred"], class_names)
    cm = compute_confusion_matrix(preds["y_true"], preds["y_pred"], class_names)

    return {
        **preds,
        "metrics": metrics,
        "confusion_matrix": cm,
    }


def save_evaluation_report(
    metrics: dict[str, Any],
    cm: np.ndarray,
    class_names: list[str],
    output_dir: str | Path = "reports/evaluation",
) -> None:
    """Save evaluation metrics and confusion matrix plot.

    Parameters
    ----------
    metrics : dict
        Output from :func:`compute_metrics`.
    cm : ndarray
        Confusion matrix.
    class_names : list of str
        Class names.
    output_dir : str or Path
        Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_to_save = {
        "overall": metrics["overall"],
        "per_class": metrics["per_class"],
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_to_save, indent=2), encoding="utf-8")
    logger.info("Metrics saved: %s", metrics_path)

    # Save classification report
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(metrics["classification_report"], encoding="utf-8")
    logger.info("Classification report saved: %s", report_path)

    # Save confusion matrix plot
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _class_names_from_checkpoint(path: str | Path) -> list[str] | None:
    """Read ordered class names from a checkpoint's metadata.class_mapping.

    Returns ``None`` when the checkpoint carries no usable mapping (older
    tomato/potato checkpoints fall back to the module-level default names,
    which match their training order anyway).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mapping = (checkpoint.get("metadata") or {}).get("class_mapping")
    if not isinstance(mapping, dict) or not mapping:
        return None
    names: list[str | None] = [None] * len(mapping)
    for key, name in mapping.items():
        names[int(key)] = str(name)
    if any(n is None for n in names):
        return None
    return names  # type: ignore[return-value]


def main() -> None:
    """CLI entry point for ``python -m src.evaluate``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="AgriMind AI — Model Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default="models/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--splits-dir", default="data/splits",
        help="Directory containing split CSVs",
    )
    parser.add_argument(
        "--output-dir", default="reports/evaluation",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size",
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image size",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    model = load_checkpoint(args.checkpoint, device=device)

    # Class names come from the checkpoint's training metadata when present
    # (crop-specific, e.g. apple); otherwise the tomato/potato defaults.
    class_names = _class_names_from_checkpoint(args.checkpoint)
    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

    # Create data loader
    from src.dataset import TomatoLeafDataset, get_eval_transforms

    dataset = TomatoLeafDataset(
        Path(args.splits_dir) / f"{args.split}.csv",
        transform=get_eval_transforms(args.image_size),
        image_size=args.image_size,
        class_to_idx={name: i for i, name in enumerate(class_names)},
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    result = evaluate(model, loader, device=device, class_names=class_names)

    # Save reports
    save_evaluation_report(
        result["metrics"], result["confusion_matrix"], class_names,
        output_dir=args.output_dir,
    )

    print(result["metrics"]["classification_report"])


if __name__ == "__main__":
    main()
