"""
AgriMind AI — Error Analysis Module (Phase 3B).

Post-training error analysis for the trained tomato leaf disease
classifier. Evaluates an existing checkpoint on a split manifest,
identifies every misclassification, computes confusion/confidence
statistics, ranks high-confidence errors, and generates
presentation-ready reports under ``reports/error_analysis/``.

Analysis only — the model, dataset, and split strategy are never
modified.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import IDX_TO_CLASS, NUM_CLASSES
from src.evaluate import predict
from src.model import load_checkpoint

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "reports/error_analysis"
ERROR_COLUMNS = [
    "image_path",
    "true_class",
    "predicted_class",
    "true_index",
    "predicted_index",
    "confidence",
    "true_class_probability",
    "margin",
]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def identify_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    image_paths: list[str],
    class_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Identify every misclassification in a set of predictions.

    Parameters
    ----------
    y_true, y_pred : array of int
        Ground-truth and predicted class indices.
    probabilities : ndarray
        Softmax probabilities of shape ``(N, num_classes)``.
    image_paths : list of str
        Image path for each sample (same order).
    class_names : list of str, optional
        Class names indexed by label.

    Returns
    -------
    list of dicts, one per misclassification, each with keys:
    ``image_path``, ``true_class``, ``predicted_class``, ``true_index``,
    ``predicted_index``, ``confidence`` (prob of predicted class),
    ``true_class_probability``, and ``margin`` (confidence minus true prob).
    """
    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities)

    n = len(y_true)
    if not (
        len(y_pred) == n
        and probabilities.shape[0] == n
        and len(image_paths) == n
    ):
        raise ValueError(
            f"Length mismatch: y_true={n}, y_pred={len(y_pred)}, "
            f"probabilities={probabilities.shape[0]}, "
            f"image_paths={len(image_paths)}"
        )

    errors: list[dict[str, Any]] = []
    for i in range(n):
        if y_true[i] == y_pred[i]:
            continue
        confidence = float(probabilities[i, y_pred[i]])
        true_prob = float(probabilities[i, y_true[i]])
        errors.append(
            {
                "image_path": str(image_paths[i]),
                "true_class": class_names[int(y_true[i])],
                "predicted_class": class_names[int(y_pred[i])],
                "true_index": int(y_true[i]),
                "predicted_index": int(y_pred[i]),
                "confidence": confidence,
                "true_class_probability": true_prob,
                "margin": confidence - true_prob,
            }
        )
    return errors


def compute_confusion_statistics(
    errors: list[dict[str, Any]],
    y_true: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute pair-level confusion statistics from misclassifications.

    Returns
    -------
    dict with ``total_errors``, ``pair_counts`` (sorted by count desc),
    ``most_confused`` (the top pair or None), and ``per_class`` error
    counts/rates.
    """
    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    y_true = np.asarray(y_true)

    pair_counter: Counter = Counter()
    for e in errors:
        pair_counter[(e["true_index"], e["predicted_index"])] += 1

    pair_counts = [
        {
            "true_class": class_names[t],
            "predicted_class": class_names[p],
            "count": c,
        }
        for (t, p), c in sorted(
            pair_counter.items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]

    per_class: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(class_names):
        support = int(np.sum(y_true == i))
        err_count = sum(1 for e in errors if e["true_index"] == i)
        per_class[name] = {
            "support": support,
            "errors": err_count,
            "error_rate": (err_count / support) if support > 0 else 0.0,
        }

    return {
        "total_errors": len(errors),
        "pair_counts": pair_counts,
        "most_confused": pair_counts[0] if pair_counts else None,
        "per_class": per_class,
    }


def compute_confidence_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute confidence statistics for correct vs incorrect predictions.

    Includes mean/median confidence per group, the confidence gap, the
    mean probability assigned to the true class on errors, and a
    reliability-style binning (accuracy per confidence bin).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities)

    n = len(y_true)
    if n == 0:
        return {
            "mean_confidence_correct": None,
            "mean_confidence_incorrect": None,
            "median_confidence_correct": None,
            "median_confidence_incorrect": None,
            "confidence_gap": None,
            "mean_true_class_probability_on_errors": None,
            "bins": [],
        }

    indices = np.arange(n)
    confidences = probabilities[indices, y_pred]
    correct_mask = y_true == y_pred

    correct_conf = confidences[correct_mask]
    wrong_conf = confidences[~correct_mask]

    mean_correct = float(correct_conf.mean()) if correct_conf.size else None
    mean_wrong = float(wrong_conf.mean()) if wrong_conf.size else None
    median_correct = (
        float(np.median(correct_conf)) if correct_conf.size else None
    )
    median_wrong = (
        float(np.median(wrong_conf)) if wrong_conf.size else None
    )
    gap = (
        (mean_correct - mean_wrong)
        if (mean_correct is not None and mean_wrong is not None)
        else None
    )

    wrong_indices = np.where(~correct_mask)[0]
    if wrong_indices.size:
        true_probs = probabilities[wrong_indices, y_true[wrong_indices]]
        mean_true_prob = float(true_probs.mean())
    else:
        mean_true_prob = None

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        accuracy = float(correct_mask[mask].mean()) if count > 0 else None
        bins.append(
            {
                "lower": float(lo),
                "upper": float(hi),
                "count": count,
                "accuracy": accuracy,
            }
        )

    return {
        "mean_confidence_correct": mean_correct,
        "mean_confidence_incorrect": mean_wrong,
        "median_confidence_correct": median_correct,
        "median_confidence_incorrect": median_wrong,
        "confidence_gap": gap,
        "mean_true_class_probability_on_errors": mean_true_prob,
        "bins": bins,
    }


def rank_high_confidence_errors(
    errors: list[dict[str, Any]], top_k: int = 10
) -> list[dict[str, Any]]:
    """Rank errors by predicted-class confidence, descending.

    High-confidence errors (the model is confidently wrong) are the most
    concerning for deployment and are surfaced first in reports.
    """
    ranked = sorted(errors, key=lambda e: e["confidence"], reverse=True)
    return ranked[:top_k] if top_k is not None else ranked


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def plot_confidence_histogram(
    correct_conf: np.ndarray,
    wrong_conf: np.ndarray,
    output_path: str | Path,
    n_bins: int = 20,
) -> Path:
    """Overlay histograms of confidence for correct vs incorrect predictions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct_conf = np.asarray(correct_conf)
    wrong_conf = np.asarray(wrong_conf)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        correct_conf, bins=bins, alpha=0.6, color="#2ecc71",
        edgecolor="black", label=f"Correct (n={correct_conf.size})",
    )
    ax.hist(
        wrong_conf, bins=bins, alpha=0.6, color="#e74c3c",
        edgecolor="black", label=f"Incorrect (n={wrong_conf.size})",
    )
    ax.set_xlabel("Confidence (predicted-class probability)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs Incorrect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Confidence histogram saved: %s", output_path)
    return output_path


def plot_confusion_pairs(
    pair_counts: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Horizontal bar chart of confused class pairs (True -> Predicted)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(10, max(3, 0.6 * len(pair_counts) + 1.5))
    )
    if not pair_counts:
        ax.text(
            0.5, 0.5, "No misclassifications",
            ha="center", va="center", transform=ax.transAxes, fontsize=14,
        )
        ax.set_axis_off()
    else:
        labels = [
            f"{p['true_class']} -> {p['predicted_class']}"
            for p in pair_counts
        ]
        counts = [p["count"] for p in pair_counts]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, counts, color="#e67e22")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("Most Confused Class Pairs (True -> Predicted)")
        max_count = max(counts)
        for i, c in enumerate(counts):
            ax.text(
                c + max_count * 0.01, i, str(c),
                va="center", fontsize=9,
            )
        ax.set_xlim(0, max_count * 1.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion pairs plot saved: %s", output_path)
    return output_path


def plot_error_grid(
    errors: list[dict[str, Any]],
    output_path: str | Path,
    max_images: int = 12,
    ncols: int = 4,
) -> Path | None:
    """Grid of the top misclassified images with true/predicted labels.

    Tolerates unreadable/missing image files (renders a gray placeholder).
    Returns the output path, or None when there are no errors.
    """
    if not errors:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = errors[:max_images]
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 4.2 * nrows)
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, err in enumerate(selected):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        loaded = False
        try:
            img = Image.open(err["image_path"]).convert("RGB")
            ax.imshow(np.asarray(img))
            loaded = True
        except Exception as exc:
            logger.debug("Could not load %s: %s", err["image_path"], exc)
        if not loaded:
            ax.set_facecolor("#bbbbbb")
        ax.set_title(
            f"True: {err['true_class']}\n"
            f"Pred: {err['predicted_class']} ({err['confidence']:.2f})",
            fontsize=9,
        )
        ax.set_axis_off()

    for idx in range(len(selected), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_axis_off()

    fig.suptitle("Top High-Confidence Misclassifications", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Error grid saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_misclassified_csv(
    errors: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write all misclassifications to CSV, sorted by confidence desc."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(errors, key=lambda e: e["confidence"], reverse=True)
    if ordered:
        df = pd.DataFrame(ordered)[ERROR_COLUMNS]
    else:
        df = pd.DataFrame(columns=ERROR_COLUMNS)
    df.to_csv(output_path, index=False)
    logger.info(
        "Misclassified CSV saved: %s (%d rows)", output_path, len(ordered)
    )
    return output_path


def generate_report(analysis: dict[str, Any], output_path: str | Path) -> Path:
    """Generate a human-readable error analysis report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conf = analysis["confidence_statistics"]
    confusion = analysis["confusion_statistics"]

    def _fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.4f}"

    lines = [
        "=" * 60,
        "AgriMind AI - Error Analysis Report",
        "=" * 60,
        "",
        f"Generated: {analysis['generated_at']}",
    ]
    if analysis.get("checkpoint"):
        lines.append(f"Checkpoint: {analysis['checkpoint']}")
    if analysis.get("split"):
        lines.append(f"Split: {analysis['split']}")

    total = analysis["total_samples"]
    n_err = analysis["total_errors"]
    lines.extend(
        [
            "",
            "--- Overview ---",
            f"Total samples: {total}",
            f"Correct predictions: {total - n_err}",
            f"Misclassified: {n_err}",
            f"Accuracy: {analysis['accuracy'] * 100:.2f}%",
            f"Error rate: {analysis['error_rate'] * 100:.2f}%",
        ]
    )

    lines.extend(["", "--- Confusion Statistics ---"])
    mc = confusion["most_confused"]
    if mc:
        lines.append(
            f"Most confused pair: {mc['true_class']} -> "
            f"{mc['predicted_class']} ({mc['count']} images)"
        )
    else:
        lines.append("Most confused pair: none (no misclassifications)")

    if confusion["pair_counts"]:
        lines.extend(["", "Confused pairs (True -> Predicted):"])
        for p in confusion["pair_counts"]:
            lines.append(
                f"  {p['true_class']} -> {p['predicted_class']}: {p['count']}"
            )

    lines.extend(["", "Per-class error rates:"])
    for name, stats in confusion["per_class"].items():
        lines.append(
            f"  {name}: {stats['errors']}/{stats['support']} errors "
            f"({stats['error_rate'] * 100:.1f}%)"
        )

    lines.extend(["", "--- Confidence Statistics ---"])
    lines.append(
        f"Mean confidence (correct):   {_fmt(conf['mean_confidence_correct'])}"
    )
    lines.append(
        f"Mean confidence (incorrect): {_fmt(conf['mean_confidence_incorrect'])}"
    )
    lines.append(
        f"Confidence gap:              {_fmt(conf['confidence_gap'])}"
    )
    lines.append(
        "Mean probability on true class (errors): "
        f"{_fmt(conf['mean_true_class_probability_on_errors'])}"
    )

    populated = [b for b in conf["bins"] if b["count"] > 0]
    if populated:
        lines.extend(["", "Confidence bins:"])
        for b in populated:
            acc = "n/a" if b["accuracy"] is None else f"{b['accuracy']:.3f}"
            lines.append(
                f"  [{b['lower']:.1f}-{b['upper']:.1f}]: "
                f"{b['count']} samples, accuracy {acc}"
            )

    top = analysis["high_confidence_errors"]
    lines.extend(
        ["", f"--- Top High-Confidence Errors (rank {len(top)}) ---"]
    )
    if not top:
        lines.append("  None - no misclassifications.")
    for rank, e in enumerate(top, start=1):
        lines.extend(
            [
                f"  {rank}. {Path(e['image_path']).name}",
                f"     True: {e['true_class']} | "
                f"Pred: {e['predicted_class']} | "
                f"Conf: {e['confidence']:.3f} | "
                f"Margin: {e['margin']:.3f}",
            ]
        )

    lines.extend(["", "--- Output Files ---"])
    for name, path in analysis.get("output_files", {}).items():
        lines.append(f"  {name}: {path}")

    lines.extend(["", "=" * 60])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Error analysis report saved: %s", output_path)
    return output_path


def generate_summary_json(
    analysis: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Generate a machine-readable JSON summary.

    The full error list is excluded (it lives in misclassified.csv);
    the top high-confidence errors are included.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in analysis.items() if k != "errors"}
    output_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Error analysis summary saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_error_analysis(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_k: int = 10,
    max_grid_images: int = 12,
    generate_grid: bool = True,
    class_names: list[str] | None = None,
    n_bins: int = 10,
    checkpoint: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Run the full error analysis pipeline.

    Parameters
    ----------
    model : nn.Module
        Trained model (eval mode is enforced by the prediction pass).
    loader : DataLoader
        Evaluation loader (use the test split -- never train).
    device : str or torch.device
        Device for inference.
    output_dir : str or Path
        Output directory for all reports/plots.
    top_k : int
        Number of high-confidence errors to rank and report.
    max_grid_images : int
        Maximum images shown in the error grid.
    generate_grid : bool
        Whether to render the misclassified-image grid.
    class_names : list of str, optional
        Class names (defaults to the project mapping).
    n_bins : int
        Number of confidence bins for the reliability breakdown.
    checkpoint, split : str, optional
        Annotation-only metadata for the reports.

    Returns
    -------
    dict with the complete analysis (errors, statistics, output paths).
    """
    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    output_dir = Path(output_dir)

    logger.info("=" * 60)
    logger.info("AgriMind AI - Error Analysis")
    logger.info("=" * 60)

    logger.info("Collecting predictions ...")
    preds = predict(model, loader, device=device)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    probabilities = preds["probabilities"]
    image_paths = preds["image_paths"]

    errors = identify_errors(
        y_true, y_pred, probabilities, image_paths, class_names
    )
    confusion_stats = compute_confusion_statistics(
        errors, y_true, class_names
    )
    confidence_stats = compute_confidence_statistics(
        y_true, y_pred, probabilities, n_bins=n_bins
    )
    top_errors = rank_high_confidence_errors(errors, top_k=top_k)

    total = int(len(y_true))
    n_err = len(errors)
    accuracy = ((total - n_err) / total) if total > 0 else 0.0
    error_rate = (n_err / total) if total > 0 else 0.0

    # Confidences for the histogram (correct vs incorrect)
    if total > 0:
        indices = np.arange(total)
        all_conf = probabilities[indices, y_pred]
        correct_mask = y_true == y_pred
        correct_conf = all_conf[correct_mask]
        wrong_conf = all_conf[~correct_mask]
    else:
        correct_conf = np.array([])
        wrong_conf = np.array([])

    analysis: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint,
        "split": split,
        "class_names": class_names,
        "total_samples": total,
        "accuracy": accuracy,
        "total_errors": n_err,
        "error_rate": error_rate,
        "errors": errors,
        "confusion_statistics": confusion_stats,
        "confidence_statistics": confidence_stats,
        "high_confidence_errors": top_errors,
        "output_files": {},
    }

    # Pre-register deterministic output paths so reports list them all
    files: dict[str, str] = {
        "misclassified_csv": str(output_dir / "misclassified.csv"),
        "confidence_histogram": str(
            output_dir / "confidence_histogram.png"
        ),
        "confusion_pairs": str(output_dir / "confusion_pairs.png"),
        "report": str(output_dir / "error_analysis_report.txt"),
        "summary_json": str(output_dir / "error_analysis.json"),
    }
    if generate_grid and errors:
        files["error_grid"] = str(output_dir / "error_grid.png")
    analysis["output_files"] = files

    write_misclassified_csv(errors, files["misclassified_csv"])
    plot_confidence_histogram(
        correct_conf, wrong_conf, files["confidence_histogram"]
    )
    plot_confusion_pairs(
        confusion_stats["pair_counts"], files["confusion_pairs"]
    )
    if "error_grid" in files:
        plot_error_grid(
            top_errors, files["error_grid"], max_images=max_grid_images
        )
    generate_report(analysis, files["report"])
    generate_summary_json(analysis, files["summary_json"])

    logger.info(
        "Error analysis complete: %d/%d misclassified (%.2f%%)",
        n_err, total, error_rate * 100,
    )
    return analysis


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python -m src.error_analysis``."""
    import argparse

    from src.dataset import TomatoLeafDataset, get_eval_transforms

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="AgriMind AI - Error Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default="models/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="Which split to analyze",
    )
    parser.add_argument(
        "--splits-dir", default="data/splits",
        help="Directory containing split CSVs",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
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
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of high-confidence errors to rank/report",
    )
    parser.add_argument(
        "--max-grid-images", type=int, default=12,
        help="Max images in the error grid",
    )
    parser.add_argument(
        "--no-grid", action="store_true",
        help="Skip error grid generation",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    model = load_checkpoint(args.checkpoint, device=device)

    manifest = Path(args.splits_dir) / f"{args.split}.csv"
    dataset = TomatoLeafDataset(
        manifest,
        transform=get_eval_transforms(args.image_size),
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    analysis = run_error_analysis(
        model,
        loader,
        device=device,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_grid_images=args.max_grid_images,
        generate_grid=not args.no_grid,
        checkpoint=args.checkpoint,
        split=args.split,
    )

    print("=" * 40)
    print("Error Analysis Summary")
    print("=" * 40)
    print(f"Total samples: {analysis['total_samples']}")
    print(
        f"Misclassified: {analysis['total_errors']} "
        f"({analysis['error_rate'] * 100:.2f}%)"
    )
    mc = analysis["confusion_statistics"]["most_confused"]
    if mc:
        print(
            f"Most confused pair: {mc['true_class']} -> "
            f"{mc['predicted_class']} ({mc['count']})"
        )
    if analysis["high_confidence_errors"]:
        top_err = analysis["high_confidence_errors"][0]
        print(
            f"Highest-confidence error: "
            f"{Path(top_err['image_path']).name} "
            f"(true {top_err['true_class']}, "
            f"pred {top_err['predicted_class']}, "
            f"conf {top_err['confidence']:.3f})"
        )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
