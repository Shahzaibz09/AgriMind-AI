"""
AgriMind AI — Exploratory Data Analysis (EDA) Module.

Produces dataset-level statistics, image property analysis, and visualizations
for the prepared tomato leaf disease classification dataset.

Outputs are saved to ``reports/eda/`` with deterministic filenames.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight"]
SPLIT_NAMES = ["train", "val", "test"]
DEFAULT_CONFIG_PATH = "configs/config.yaml"
DEFAULT_REPORTS_DIR = "reports/eda"


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifests(splits_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load train/val/test split CSVs from *splits_dir*.

    Each CSV must have columns ``image_path`` and ``clean_class``.
    Returns a dict ``{"train": df, "val": df, "test": df}``.
    """
    splits_dir = Path(splits_dir)
    manifests: dict[str, pd.DataFrame] = {}
    for split in SPLIT_NAMES:
        path = splits_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split manifest not found: {path}")
        df = pd.read_csv(path)
        if not {"image_path", "clean_class"}.issubset(df.columns):
            raise ValueError(
                f"Manifest {path} missing required columns: image_path, clean_class"
            )
        manifests[split] = df
    return manifests


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------


def compute_dataset_stats(
    manifests: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Compute overall dataset statistics from split manifests.

    Returns a dict with total images, per-class counts, per-split counts,
    and class distribution percentages.
    """
    all_dfs = list(manifests.values())
    combined = pd.concat(all_dfs, ignore_index=True)
    total = len(combined)

    class_counts = combined["clean_class"].value_counts().to_dict()
    split_counts = {name: len(df) for name, df in manifests.items()}

    class_pct = {
        cls: (count / total * 100 if total > 0 else 0.0)
        for cls, count in class_counts.items()
    }

    return {
        "total_images": total,
        "class_counts": class_counts,
        "split_counts": split_counts,
        "class_percentages": class_pct,
    }


def compute_split_class_distribution(
    manifests: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute a cross-tabulation of split × class counts."""
    rows = []
    for split_name, df in manifests.items():
        counts = df["clean_class"].value_counts().to_dict()
        row = {"split": split_name, **counts}
        rows.append(row)
    dist_df = pd.DataFrame(rows).set_index("split").fillna(0).astype(int)
    return dist_df


# ---------------------------------------------------------------------------
# Image property analysis
# ---------------------------------------------------------------------------


def analyze_image_properties(
    manifests: dict[str, pd.DataFrame],
    sample_size: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze image properties across the dataset.

    If *sample_size* is provided, only analyze a random sample of images.
    Returns width, height, aspect ratio, mode, format, min/max dimensions,
    and common dimensions.
    """
    all_paths = []
    for df in manifests.values():
        all_paths.extend(df["image_path"].tolist())

    if sample_size is not None and sample_size < len(all_paths):
        rng = random.Random(seed)
        all_paths = rng.sample(all_paths, sample_size)

    widths: list[int] = []
    heights: list[int] = []
    modes: list[str] = []
    formats: list[str] = []
    invalid: list[str] = []

    for p in all_paths:
        try:
            with Image.open(p) as img:
                widths.append(img.size[0])
                heights.append(img.size[1])
                modes.append(img.mode)
                formats.append(img.format or Path(p).suffix.lstrip(".").upper())
        except Exception as e:
            logger.warning("Invalid/unreadable image %s: %s", p, e)
            invalid.append(str(p))

    if not widths:
        return {"error": "No valid images found", "invalid_count": len(invalid)}

    aspect_ratios = [w / h for w, h in zip(widths, heights)]
    dim_counter = Counter(zip(widths, heights))
    common_dims = dim_counter.most_common(5)

    return {
        "analyzed_count": len(widths),
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios,
        "modes": modes,
        "formats": formats,
        "invalid_images": invalid,
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "mean_width": float(np.mean(widths)),
        "mean_height": float(np.mean(heights)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        "common_dimensions": [(w, h, c) for (w, h), c in common_dims],
        "mode_counts": Counter(modes),
        "format_counts": Counter(formats),
    }


def detect_invalid_images(manifests: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    """Detect invalid/unreadable images without modifying them.

    Returns a list of dicts with ``path`` and ``error`` keys.
    """
    invalid: list[dict[str, str]] = []
    for df in manifests.values():
        for p in df["image_path"]:
            try:
                with Image.open(p) as img:
                    img.verify()
            except Exception as e:
                invalid.append({"path": str(p), "error": str(e)})
    return invalid


# ---------------------------------------------------------------------------
# Pixel statistics
# ---------------------------------------------------------------------------


def compute_pixel_statistics(
    manifests: dict[str, pd.DataFrame],
    sample_size: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute per-channel pixel statistics from a random sample of images.

    Returns mean and std per channel (R, G, B).
    """
    all_paths = []
    for df in manifests.values():
        all_paths.extend(df["image_path"].tolist())

    rng = random.Random(seed)
    if sample_size < len(all_paths):
        sample_paths = rng.sample(all_paths, sample_size)
    else:
        sample_paths = all_paths

    channel_means: list[list[float]] = [[], [], []]

    for p in sample_paths:
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                arr = np.array(img, dtype=np.float32)
                for c in range(3):
                    channel_means[c].append(float(arr[:, :, c].mean()))
        except Exception as e:
            logger.warning("Skipping pixel stats for %s: %s", p, e)

    if not channel_means[0]:
        return {"error": "No valid images for pixel statistics"}

    return {
        "sampled_count": len(channel_means[0]),
        "channel_mean": {
            "R": float(np.mean(channel_means[0])),
            "G": float(np.mean(channel_means[1])),
            "B": float(np.mean(channel_means[2])),
        },
        "channel_std": {
            "R": float(np.std(channel_means[0])),
            "G": float(np.std(channel_means[1])),
            "B": float(np.std(channel_means[2])),
        },
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def generate_visualizations(
    stats: dict[str, Any],
    split_class_dist: pd.DataFrame,
    img_props: dict[str, Any],
    manifests: dict[str, pd.DataFrame],
    output_dir: Path,
    seed: int = 42,
) -> list[Path]:
    """Generate EDA visualizations and save to *output_dir*.

    Returns a list of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # 1. Overall class distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = list(stats["class_counts"].keys())
    counts = [stats["class_counts"][c] for c in classes]
    bars = ax.bar(classes, counts, color=["#2ecc71", "#e74c3c", "#3498db"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Overall Class Distribution")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha="center", va="bottom")
    fig.tight_layout()
    p = output_dir / "class_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    generated.append(p)

    # 2. Split distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    split_names = list(stats["split_counts"].keys())
    split_counts = [stats["split_counts"][s] for s in split_names]
    ax.pie(split_counts, labels=split_names, autopct="%1.1f%%",
           colors=["#3498db", "#e67e22", "#9b59b6"])
    ax.set_title("Split Distribution")
    fig.tight_layout()
    p = output_dir / "split_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    generated.append(p)

    # 3. Class distribution by split (grouped bar chart)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(split_class_dist.index))
    width = 0.25
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    for i, col in enumerate(split_class_dist.columns):
        ax.bar(x + i * width, split_class_dist[col], width, label=col, color=colors[i % len(colors)])
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution by Split")
    ax.set_xticks(x + width)
    ax.set_xticklabels(split_class_dist.index)
    ax.legend()
    fig.tight_layout()
    p = output_dir / "class_by_split.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    generated.append(p)

    # 4. Image dimension distribution
    if "widths" in img_props and img_props["widths"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(img_props["widths"], bins=30, color="#3498db", alpha=0.7)
        axes[0].set_xlabel("Width (px)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Image Width Distribution")
        axes[1].hist(img_props["heights"], bins=30, color="#e67e22", alpha=0.7)
        axes[1].set_xlabel("Height (px)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Image Height Distribution")
        fig.tight_layout()
        p = output_dir / "dimension_distribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        generated.append(p)

    # 5. Representative sample grid for all three classes
    rng = random.Random(seed)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for row, cls in enumerate(CLASS_NAMES):
        # Collect paths for this class from all splits
        cls_paths = []
        for df in manifests.values():
            cls_df = df[df["clean_class"] == cls]
            cls_paths.extend(cls_df["image_path"].tolist())
        if cls_paths:
            sample = rng.sample(cls_paths, min(5, len(cls_paths)))
            for col, path in enumerate(sample):
                try:
                    img = Image.open(path).convert("RGB")
                    axes[row, col].imshow(np.array(img))
                    axes[row, col].axis("off")
                    if col == 0:
                        axes[row, col].set_title(cls, fontsize=10, loc="left")
                except Exception:
                    axes[row, col].axis("off")
        # Fill remaining columns if fewer than 5 samples
        for col in range(len(sample) if cls_paths else 0, 5):
            axes[row, col].axis("off")
    fig.suptitle("Representative Samples by Class", fontsize=12)
    fig.tight_layout()
    p = output_dir / "sample_grid.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    generated.append(p)

    return generated


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    stats: dict[str, Any],
    split_class_dist: pd.DataFrame,
    img_props: dict[str, Any],
    pixel_stats: dict[str, Any],
    invalid_images: list[dict[str, str]],
    viz_paths: list[Path],
    output_path: Path,
) -> None:
    """Generate a human-readable EDA report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 60,
        "AgriMind AI — Exploratory Data Analysis Report",
        "=" * 60,
        "",
        f"Generated: {timestamp}",
        "",
        "--- Dataset Overview ---",
        f"Total images: {stats['total_images']}",
        "",
        "Images per class:",
    ]
    for cls, count in sorted(stats["class_counts"].items()):
        pct = stats["class_percentages"].get(cls, 0.0)
        lines.append(f"  {cls}: {count} ({pct:.1f}%)")

    lines.extend(["", "Images per split:"])
    for split, count in stats["split_counts"].items():
        lines.append(f"  {split}: {count}")

    lines.extend(["", "--- Split x Class Distribution ---", ""])
    lines.append(split_class_dist.to_string())

    lines.extend(["", "", "--- Image Properties ---"])
    if "error" not in img_props:
        lines.extend([
            f"Analyzed: {img_props['analyzed_count']} images",
            f"Width: min={img_props['min_width']}, max={img_props['max_width']}, mean={img_props['mean_width']:.1f}",
            f"Height: min={img_props['min_height']}, max={img_props['max_height']}, mean={img_props['mean_height']:.1f}",
            f"Aspect ratio (mean): {img_props['mean_aspect_ratio']:.3f}",
            "",
            "Mode counts:",
        ])
        for mode, count in img_props["mode_counts"].most_common():
            lines.append(f"  {mode}: {count}")
        lines.extend(["", "Format counts:"])
        for fmt, count in img_props["format_counts"].most_common():
            lines.append(f"  {fmt}: {count}")
        lines.extend(["", "Common dimensions (WxH):"])
        for w, h, c in img_props["common_dimensions"]:
            lines.append(f"  {w}x{h}: {c} images")
    else:
        lines.append(f"Error: {img_props['error']}")

    lines.extend(["", "--- Pixel Statistics ---"])
    if "error" not in pixel_stats:
        lines.extend([
            f"Sampled: {pixel_stats['sampled_count']} images",
            "Channel mean: " + ", ".join(f"{ch}={v:.2f}" for ch, v in pixel_stats["channel_mean"].items()),
            "Channel std: " + ", ".join(f"{ch}={v:.2f}" for ch, v in pixel_stats["channel_std"].items()),
        ])
    else:
        lines.append(f"Error: {pixel_stats['error']}")

    lines.extend(["", "--- Invalid/Unreadable Images ---"])
    if invalid_images:
        lines.append(f"Count: {len(invalid_images)}")
        for item in invalid_images[:20]:  # limit output
            lines.append(f"  {item['path']}: {item['error']}")
        if len(invalid_images) > 20:
            lines.append(f"  ... and {len(invalid_images) - 20} more")
    else:
        lines.append("None detected.")

    lines.extend(["", "--- Generated Visualizations ---"])
    for p in viz_paths:
        lines.append(f"  {p.name}")

    lines.extend(["", "=" * 60])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("EDA report written: %s", output_path)


def generate_summary_json(
    stats: dict[str, Any],
    split_class_dist: pd.DataFrame,
    img_props: dict[str, Any],
    pixel_stats: dict[str, Any],
    invalid_images: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Generate a machine-readable EDA summary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_images": stats["total_images"],
        "class_counts": stats["class_counts"],
        "split_counts": stats["split_counts"],
        "class_percentages": stats["class_percentages"],
        "split_class_distribution": split_class_dist.to_dict(),
        "invalid_image_count": len(invalid_images),
    }

    if "error" not in img_props:
        summary["image_properties"] = {
            "analyzed_count": img_props["analyzed_count"],
            "min_width": img_props["min_width"],
            "max_width": img_props["max_width"],
            "mean_width": img_props["mean_width"],
            "min_height": img_props["min_height"],
            "max_height": img_props["max_height"],
            "mean_height": img_props["mean_height"],
            "mean_aspect_ratio": img_props["mean_aspect_ratio"],
            "mode_counts": dict(img_props["mode_counts"]),
            "format_counts": dict(img_props["format_counts"]),
            "common_dimensions": [
                {"width": w, "height": h, "count": c}
                for w, h, c in img_props["common_dimensions"]
            ],
        }

    if "error" not in pixel_stats:
        summary["pixel_statistics"] = {
            "sampled_count": pixel_stats["sampled_count"],
            "channel_mean": pixel_stats["channel_mean"],
            "channel_std": pixel_stats["channel_std"],
        }

    output_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("EDA summary written: %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_eda(
    splits_dir: str | Path = "data/splits",
    output_dir: str | Path = DEFAULT_REPORTS_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Run the full EDA pipeline.

    Returns a summary dict with paths to generated outputs.
    """
    # Load configuration
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    seed = config.get("seed", 42)

    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)

    logger.info("=" * 60)
    logger.info("AgriMind AI — Exploratory Data Analysis")
    logger.info("=" * 60)

    # 1. Load manifests
    logger.info("Loading split manifests from %s ...", splits_dir)
    manifests = load_manifests(splits_dir)
    for name, df in manifests.items():
        logger.info("  %s: %d images", name, len(df))

    # 2. Dataset statistics
    logger.info("Computing dataset statistics ...")
    stats = compute_dataset_stats(manifests)
    logger.info("Total images: %d", stats["total_images"])

    # 3. Split x class distribution
    split_class_dist = compute_split_class_distribution(manifests)
    logger.info("Split x class distribution computed.")

    # 4. Image property analysis (sample for speed)
    logger.info("Analyzing image properties ...")
    img_props = analyze_image_properties(manifests, sample_size=1000, seed=seed)
    logger.info("  Analyzed %d images", img_props.get("analyzed_count", 0))

    # 5. Detect invalid images
    logger.info("Detecting invalid images ...")
    invalid_images = detect_invalid_images(manifests)
    logger.info("  Invalid images: %d", len(invalid_images))

    # 6. Pixel statistics
    logger.info("Computing pixel statistics ...")
    pixel_stats = compute_pixel_statistics(manifests, sample_size=500, seed=seed)
    logger.info("  Sampled %d images for pixel stats", pixel_stats.get("sampled_count", 0))

    # 7. Generate visualizations
    logger.info("Generating visualizations ...")
    viz_paths = generate_visualizations(
        stats, split_class_dist, img_props, manifests, output_dir, seed=seed
    )
    logger.info("  Generated %d visualizations", len(viz_paths))

    # 8. Generate report
    report_path = output_dir / "eda_report.txt"
    generate_report(stats, split_class_dist, img_props, pixel_stats,
                    invalid_images, viz_paths, report_path)

    # 9. Generate JSON summary
    summary_path = output_dir / "eda_summary.json"
    generate_summary_json(stats, split_class_dist, img_props, pixel_stats,
                          invalid_images, summary_path)

    logger.info("=" * 60)
    logger.info("EDA complete. Outputs in %s", output_dir)
    logger.info("=" * 60)

    return {
        "stats": stats,
        "split_class_dist": split_class_dist,
        "img_props": img_props,
        "pixel_stats": pixel_stats,
        "invalid_images": invalid_images,
        "viz_paths": viz_paths,
        "report_path": report_path,
        "summary_path": summary_path,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python -m src.eda``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="AgriMind AI — Exploratory Data Analysis")
    parser.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Directory containing train.csv, val.csv, test.csv (default: data/splits)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_REPORTS_DIR,
        help=f"Output directory for EDA reports (default: {DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()

    run_eda(splits_dir=args.splits_dir, output_dir=args.output_dir, config_path=args.config)


if __name__ == "__main__":
    main()
