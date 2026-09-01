"""
AgriMind AI — Grad-CAM Error Investigation (Phase 3B follow-up).

Batch Grad-CAM analysis for misclassified test images. For each error,
generates dual explanations: one for the predicted (wrong) class and one
for the true class, revealing what the model "sees" in each case.

No existing files are modified — this module reuses the GradCAM engine,
checkpoint loader, and dataset transforms from existing modules.
"""

from __future__ import annotations

import json
import logging
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

from src.dataset import (
    IDX_TO_CLASS,
    NUM_CLASSES,
    get_eval_transforms,
)
from src.gradcam import GradCAM, denormalize, generate_overlay
from src.model import load_checkpoint

logger = logging.getLogger(__name__)

DEFAULT_ERRORS_CSV = "reports/error_analysis/misclassified.csv"
DEFAULT_OUTPUT_DIR = "reports/gradcam_errors"


# ---------------------------------------------------------------------------
# Single-image dual Grad-CAM
# ---------------------------------------------------------------------------


def analyze_error_gradcam(
    model: nn.Module,
    image_path: str | Path,
    true_index: int,
    predicted_index: int,
    device: str | torch.device = "cpu",
    image_size: int = 224,
    alpha: float = 0.4,
) -> dict[str, Any]:
    """Run dual Grad-CAM on a single misclassified image.

    Generates heatmaps for both the predicted (wrong) class and the true
    class, revealing what the model focuses on for each.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.
    image_path : str or Path
        Path to the image file.
    true_index : int
        Ground-truth class index.
    predicted_index : int
        Predicted (wrong) class index.
    device : str or torch.device
        Device for inference.
    image_size : int
        Target image size.
    alpha : float
        Overlay blend factor.

    Returns
    -------
    dict with ``pred_cam``, ``true_cam``, ``pred_overlay``,
    ``true_overlay``, ``original``, ``probabilities``, and metadata.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    model.eval()

    transform = get_eval_transforms(image_size)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Pass 1: Grad-CAM for the predicted (wrong) class
    with GradCAM(model) as cam_engine:
        cam_pred, _, logits = cam_engine(img_tensor, class_idx=predicted_index)

    # Pass 2: Grad-CAM for the true class
    with GradCAM(model) as cam_engine:
        cam_true, _, _ = cam_engine(img_tensor, class_idx=true_index)

    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    prob_dict = {IDX_TO_CLASS[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    original = denormalize(img_tensor.squeeze(0).cpu())
    pred_overlay = generate_overlay(original, cam_pred, alpha=alpha)
    true_overlay = generate_overlay(original, cam_true, alpha=alpha)

    return {
        "image_path": str(image_path),
        "true_index": true_index,
        "predicted_index": predicted_index,
        "true_class": IDX_TO_CLASS[true_index],
        "predicted_class": IDX_TO_CLASS[predicted_index],
        "probabilities": prob_dict,
        "confidence": float(probs[predicted_index]),
        "pred_cam": cam_pred,
        "true_cam": cam_true,
        "original": original,
        "pred_overlay": pred_overlay,
        "true_overlay": true_overlay,
    }


# ---------------------------------------------------------------------------
# Save individual image outputs
# ---------------------------------------------------------------------------


def _save_image_outputs(
    result: dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> dict[str, str]:
    """Save heatmap and overlay PNGs for one error analysis result.

    Returns dict mapping key names to file paths.
    """
    paths: dict[str, str] = {}

    # Pred heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(result["pred_cam"], cmap="jet")
    ax.axis("off")
    ax.set_title(
        f"Grad-CAM: {result['predicted_class']} "
        f"({result['confidence']:.3f})"
    )
    fig.tight_layout()
    pred_hm = output_dir / f"{prefix}_pred_heatmap.png"
    fig.savefig(pred_hm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["pred_heatmap"] = str(pred_hm)

    # True heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(result["true_cam"], cmap="jet")
    ax.axis("off")
    ax.set_title(f"Grad-CAM: {result['true_class']} (true)")
    fig.tight_layout()
    true_hm = output_dir / f"{prefix}_true_heatmap.png"
    fig.savefig(true_hm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["true_heatmap"] = str(true_hm)

    # Pred overlay
    Image.fromarray(result["pred_overlay"]).save(
        output_dir / f"{prefix}_pred_overlay.png"
    )
    paths["pred_overlay"] = str(output_dir / f"{prefix}_pred_overlay.png")

    # True overlay
    Image.fromarray(result["true_overlay"]).save(
        output_dir / f"{prefix}_true_overlay.png"
    )
    paths["true_overlay"] = str(output_dir / f"{prefix}_true_overlay.png")

    return paths


# ---------------------------------------------------------------------------
# Summary grid
# ---------------------------------------------------------------------------


def generate_error_grid(
    results_by_pair: dict[str, list[dict[str, Any]]],
    output_path: str | Path,
    top_per_pair: int = 3,
) -> Path:
    """Generate a presentation-ready summary grid.

    One row per confusion pair, showing the top-N highest-confidence
    errors with original, predicted-class overlay, and true-class overlay.

    Parameters
    ----------
    results_by_pair : dict
        Mapping from pair label (e.g. ``"Early Blight_to_Late Blight"``)
        to list of result dicts (sorted by confidence descending).
    output_path : str or Path
        Output PNG path.
    top_per_pair : int
        Max images shown per pair row.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pair_labels = list(results_by_pair.keys())
    n_rows = len(pair_labels)

    if n_rows == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "No misclassifications to visualize",
            ha="center", va="center", fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Empty error grid saved: %s", output_path)
        return output_path

    n_cols = top_per_pair * 3  # original, pred overlay, true overlay

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    for row_idx, pair_label in enumerate(pair_labels):
        items = results_by_pair[pair_label][:top_per_pair]
        for col_group, result in enumerate(items):
            base_col = col_group * 3

            # Original
            ax = axes[row_idx][base_col]
            ax.imshow(result["original"])
            ax.set_title(
                f"Original\nTrue: {result['true_class']}", fontsize=8
            )
            ax.axis("off")

            # Pred overlay
            ax = axes[row_idx][base_col + 1]
            ax.imshow(result["pred_overlay"])
            ax.set_title(
                f"Pred: {result['predicted_class']}\n"
                f"Conf: {result['confidence']:.3f}",
                fontsize=8,
            )
            ax.axis("off")

            # True overlay
            ax = axes[row_idx][base_col + 2]
            ax.imshow(result["true_overlay"])
            ax.set_title(f"True: {result['true_class']}", fontsize=8)
            ax.axis("off")

        # Pair label on the left
        axes[row_idx][0].set_ylabel(
            pair_label.replace("_", " -> "),
            fontsize=10, fontweight="bold", rotation=0,
            labelpad=120, va="center",
        )

    fig.suptitle(
        "Grad-CAM Error Investigation: Predicted vs True Class",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Error grid saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_error_report(
    all_results: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate human-readable TXT and machine-readable JSON summaries.

    Returns dict with ``report`` and ``summary_json`` paths.
    """
    paths: dict[str, Path] = {}

    # --- JSON summary ---
    summary: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_errors_analyzed": len(all_results),
        "errors": [],
    }
    for r in all_results:
        summary["errors"].append(
            {
                "image_path": r["image_path"],
                "true_class": r["true_class"],
                "predicted_class": r["predicted_class"],
                "confidence": r["confidence"],
                "probabilities": r["probabilities"],
                "output_files": r.get("output_files", {}),
            }
        )

    json_path = output_dir / "gradcam_errors_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    paths["summary_json"] = json_path

    # --- TXT report ---
    lines = [
        "=" * 60,
        "AgriMind AI - Grad-CAM Error Investigation Report",
        "=" * 60,
        "",
        f"Generated: {summary['generated_at']}",
        f"Total errors analyzed: {len(all_results)}",
        "",
    ]

    # Group by pair
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for r in all_results:
        pair_key = f"{r['true_class']} -> {r['predicted_class']}"
        by_pair.setdefault(pair_key, []).append(r)

    for pair_key, items in by_pair.items():
        lines.append(f"--- {pair_key} ({len(items)} images) ---")
        for idx, r in enumerate(items, start=1):
            lines.append(
                f"  {idx}. {Path(r['image_path']).name}"
                f"  (conf {r['confidence']:.3f})"
            )
        lines.append("")

    lines.extend(["--- Output Files ---", ""])
    for r in all_results:
        stem = Path(r["image_path"]).stem
        lines.append(f"  {stem}:")
        for key, path in r.get("output_files", {}).items():
            lines.append(f"    {key}: {path}")
    lines.extend(["", "=" * 60])

    txt_path = output_dir / "gradcam_errors_report.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["report"] = txt_path

    logger.info("Reports saved: %s", list(paths.values()))
    return paths


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def _pair_dir_name(true_class: str, predicted_class: str) -> str:
    """Convert a confusion pair to a directory-safe name."""
    return f"{true_class.replace(' ', '_')}_to_{predicted_class.replace(' ', '_')}"


def run_gradcam_on_errors(
    model: nn.Module,
    errors_csv: str | Path = DEFAULT_ERRORS_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str | torch.device = "cpu",
    image_size: int = 224,
    alpha: float = 0.4,
    top_k: int = 0,
    top_per_pair_grid: int = 3,
) -> dict[str, Any]:
    """Run Grad-CAM analysis on all misclassified images.

    Parameters
    ----------
    model : nn.Module
        Trained model (loaded checkpoint).
    errors_csv : str or Path
        Path to the misclassified CSV from error analysis.
    output_dir : str or Path
        Output directory for all Grad-CAM outputs.
    device : str or torch.device
        Device for inference.
    image_size : int
        Target image size.
    alpha : float
        Overlay blend factor.
    top_k : int
        If > 0, only analyze the top-k errors by confidence.
        If 0 (default), analyze all errors.
    top_per_pair_grid : int
        Max images per pair row in the summary grid.

    Returns
    -------
    dict with all results and output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_csv = Path(errors_csv)

    logger.info("=" * 60)
    logger.info("AgriMind AI - Grad-CAM Error Investigation")
    logger.info("=" * 60)

    # Read error list
    df = pd.read_csv(errors_csv)
    logger.info("Loaded %d errors from %s", len(df), errors_csv)

    if top_k > 0:
        df = df.head(top_k)
        logger.info("Analyzing top %d errors", top_k)

    # Group by confusion pair for directory organisation
    if len(df) > 0:
        df["_pair_dir"] = df.apply(
            lambda r: _pair_dir_name(r["true_class"], r["predicted_class"]),
            axis=1,
        )
    else:
        df["_pair_dir"] = pd.Series(dtype=str)

    all_results: list[dict[str, Any]] = []
    results_by_pair: dict[str, list[dict[str, Any]]] = {}

    model.to(device)
    model.eval()

    for idx, row in df.iterrows():
        img_path = row["image_path"]
        true_idx = int(row["true_index"])
        pred_idx = int(row["predicted_index"])
        pair_dir_name = row["_pair_dir"]

        logger.info(
            "[%d/%d] %s (true=%s, pred=%s)",
            idx + 1, len(df),
            Path(img_path).name,
            IDX_TO_CLASS[true_idx],
            IDX_TO_CLASS[pred_idx],
        )

        try:
            result = analyze_error_gradcam(
                model, img_path,
                true_index=true_idx,
                predicted_index=pred_idx,
                device=device,
                image_size=image_size,
                alpha=alpha,
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", img_path, exc)
            continue

        # Create pair subdirectory
        pair_dir = output_dir / pair_dir_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Number images by confidence rank within each pair
        pair_results = results_by_pair.setdefault(pair_dir_name, [])
        rank = len(pair_results) + 1
        stem = Path(img_path).stem
        prefix = f"{rank:02d}_{stem}"

        # Save individual outputs
        output_files = _save_image_outputs(result, pair_dir, prefix)
        result["output_files"] = output_files
        result["pair_key"] = pair_dir_name
        result["rank_in_pair"] = rank

        pair_results.append(result)
        all_results.append(result)

    # Summary grid
    if all_results:
        grid_path = generate_error_grid(
            results_by_pair,
            output_dir / "summary_grid.png",
            top_per_pair=top_per_pair_grid,
        )
    else:
        grid_path = None

    # Reports
    report_paths = generate_error_report(all_results, output_dir)

    analysis: dict[str, Any] = {
        "total_analyzed": len(all_results),
        "results": all_results,
        "results_by_pair": results_by_pair,
        "grid_path": str(grid_path) if grid_path else None,
        "report_paths": {k: str(v) for k, v in report_paths.items()},
    }

    logger.info(
        "Grad-CAM error investigation complete: %d images analyzed",
        len(all_results),
    )
    return analysis


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python -m src.gradcam_errors``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="AgriMind AI - Grad-CAM Error Investigation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--errors-csv", default=DEFAULT_ERRORS_CSV,
        help="Path to misclassified CSV from error analysis",
    )
    parser.add_argument(
        "--checkpoint", default="models/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Output directory for Grad-CAM error outputs",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image size",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.4,
        help="Overlay blend factor",
    )
    parser.add_argument(
        "--top-k", type=int, default=0,
        help="Analyze only top-k errors by confidence (0 = all)",
    )
    parser.add_argument(
        "--top-per-pair-grid", type=int, default=3,
        help="Max images per pair row in summary grid",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    model = load_checkpoint(args.checkpoint, device=device)

    analysis = run_gradcam_on_errors(
        model,
        errors_csv=args.errors_csv,
        output_dir=args.output_dir,
        device=device,
        image_size=args.image_size,
        alpha=args.alpha,
        top_k=args.top_k,
        top_per_pair_grid=args.top_per_pair_grid,
    )

    print("=" * 40)
    print("Grad-CAM Error Investigation Summary")
    print("=" * 40)
    print(f"Errors analyzed: {analysis['total_analyzed']}")
    for pair_key, items in analysis["results_by_pair"].items():
        label = pair_key.replace("_to_", " -> ").replace("_", " ")
        print(f"  {label}: {len(items)} images")
    if analysis["grid_path"]:
        print(f"Summary grid: {analysis['grid_path']}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
