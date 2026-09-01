"""
AgriMind AI — Crop Classifier Dataset Builder.

Builds train/val/test manifests for the lightweight 3-class crop
classifier (Tomato / Potato / Apple) by sampling from the EXISTING
per-crop split manifests — no new data is downloaded and no disease
pipeline artifact is touched.

Sampling is deterministic (fixed seed) and stratified by crop: each
manifest draws the same number of images per crop from the *matching*
disease split (crop-classifier train rows come from the disease-train
rows, etc.), so the crop classifier's own splits stay disjoint by
construction.

Output manifests use the schema expected by the existing dataset /
training code (columns ``image_path`` and ``clean_class``), with
``clean_class`` holding the crop label (Tomato / Potato / Apple).

Usage::

    python -m src.build_crop_classifier_dataset
    python -m src.build_crop_classifier_dataset --train-per-crop 500
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Existing per-crop split directories (disease manifests) -> crop label.
CROP_SPLIT_DIRS: dict[str, Path] = {
    "Tomato": Path("data/splits"),
    "Potato": Path("data/splits_potato"),
    "Apple": Path("data/splits_apple"),
}

SPLIT_FILES = ("train.csv", "val.csv", "test.csv")
DEFAULT_OUTPUT_DIR = Path("data/splits_crop_classifier")


def build_manifests(
    train_per_crop: int = 500,
    val_per_crop: int = 100,
    test_per_crop: int = 100,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
) -> None:
    """Sample per-crop rows from the disease splits and write manifests.

    Parameters
    ----------
    train_per_crop, val_per_crop, test_per_crop : int
        Number of images to draw per crop for each split.
    output_dir : Path or str
        Destination directory for train.csv / val.csv / test.csv.
    seed : int
        Random seed (deterministic sampling).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_split = {"train": train_per_crop, "val": val_per_crop, "test": test_per_crop}
    report_lines: list[str] = [
        "AgriMind AI — Crop Classifier Dataset Build Report",
        "=" * 52,
        f"Seed: {seed}",
        "",
    ]

    for split, n_per_crop in per_split.items():
        frames = []
        for crop_label, splits_dir in CROP_SPLIT_DIRS.items():
            src = Path(splits_dir) / f"{split}.csv"
            if not src.exists():
                raise FileNotFoundError(f"Missing source split: {src}")
            df = pd.read_csv(src)
            if len(df) < n_per_crop:
                raise ValueError(
                    f"{src} has only {len(df)} rows; need {n_per_crop} "
                    f"for crop '{crop_label}'"
                )
            sampled = df.sample(n=n_per_crop, random_state=seed)
            sampled = sampled.assign(clean_class=crop_label)
            frames.append(sampled[["image_path", "clean_class"]])
            report_lines.append(
                f"{split:5s} {crop_label:7s}: {n_per_crop} sampled from {src}"
            )

            # Sanity: every sampled image must exist on disk.
            missing = [
                p for p in sampled["image_path"] if not Path(p).exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Sampled images missing on disk for {crop_label}: "
                    f"{missing[:3]}"
                )

        manifest = pd.concat(frames, ignore_index=True)
        out_path = output_dir / f"{split}.csv"
        manifest.to_csv(out_path, index=False)
        logger.info(
            "Wrote %s (%d rows: %s)",
            out_path,
            len(manifest),
            manifest["clean_class"].value_counts().to_dict(),
        )
        report_lines.append(
            f"{split:5s} TOTAL : {len(manifest)} rows -> {out_path}"
        )
        report_lines.append("")

    report_path = output_dir / "build_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Build report written: %s", report_path)


def main() -> None:
    """CLI entry point for ``python -m src.build_crop_classifier_dataset``."""
    parser = argparse.ArgumentParser(
        description=(
            "Build 3-class crop classifier manifests from the existing "
            "per-crop disease splits (deterministic, stratified sampling)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-per-crop", type=int, default=500,
        help="Training images sampled per crop",
    )
    parser.add_argument(
        "--val-per-crop", type=int, default=100,
        help="Validation images sampled per crop",
    )
    parser.add_argument(
        "--test-per-crop", type=int, default=100,
        help="Test images sampled per crop",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the manifests",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic sampling",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AgriMind AI — Crop Classifier Dataset Builder")
    logger.info("=" * 60)
    build_manifests(
        train_per_crop=args.train_per_crop,
        val_per_crop=args.val_per_crop,
        test_per_crop=args.test_per_crop,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
