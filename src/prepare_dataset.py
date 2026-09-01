"""
AgriMind AI — Dataset Preparation Pipeline

Scans, verifies, and splits the PlantVillage tomato or potato leaf dataset.
Produces metadata, validation reports, and stratified train/val/test manifests.

Usage:
    python -m src.prepare_dataset
    python -m src.prepare_dataset --help
    python -m src.prepare_dataset --skip-split
    python -m src.prepare_dataset --config configs/config_potato.yaml
"""

import argparse
import hashlib
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASS_MAP = {
    "Tomato___healthy": "Healthy",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
}

POTATO_CLASS_MAP = {
    "Potato___healthy": "Healthy",
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
}

APPLE_CLASS_MAP = {
    "Apple___healthy": "Healthy",
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Black Rot",
}

# Per-crop mapping of raw PlantVillage class folders -> clean class names.
# Tomato and potato share the same clean class names; apple introduces its
# own disease names, which the crop-aware ``class_to_idx`` threading in
# src.dataset / src.train / src.evaluate handles (defaults unchanged).
CROP_CLASS_MAPS = {
    "tomato": CLASS_MAP,
    "potato": POTATO_CLASS_MAP,
    "apple": APPLE_CLASS_MAP,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found at %s. Using defaults.", config_path)
        return {
            "seed": 42,
            "data": {
                "raw": "data/raw",
                "processed": "data/processed",
                "splits": "data/splits",
            },
        }
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded configuration from %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_image(file_path: Path) -> dict | None:
    """
    Validate a single image file using Pillow.

    Returns a dict with image metadata on success, or None on failure.
    Never modifies or deletes the source file.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Re-open after verify (verify can leave the image in an unusable state)
        with Image.open(file_path) as img:
            img.load()
            width, height = img.size
            mode = img.mode
        return {
            "image_path": str(file_path),
            "file_extension": file_path.suffix.lower(),
            "image_mode": mode,
            "width": width,
            "height": height,
            "md5": compute_md5(file_path),
        }
    except Exception as e:
        logger.warning("Corrupted/unreadable image: %s — %s", file_path, e)
        return None


def is_verified_leaf_id(leaf_id: str) -> bool:
    """Return True if *leaf_id* represents a verified physical leaf.

    Unverified values (``"unknown"`` or ``"fallback_*"``) are NOT treated
    as a single physical leaf — each image with such a value is its own
    effective group during splitting.
    """
    if not leaf_id:
        return False
    lower = leaf_id.lower()
    return lower != "unknown" and not lower.startswith("fallback_")


# ---------------------------------------------------------------------------
# Source metadata (leaf grouping from Hugging Face)
# ---------------------------------------------------------------------------


def load_source_metadata(source_metadata_path: Path) -> dict[str, str]:
    """
    Load source metadata CSV from the acquisition script.

    Expected columns: image_path, leaf_id, label.
    Returns a dict mapping normalised image_path -> leaf_id.
    Returns an empty dict if the file does not exist.
    """
    if not source_metadata_path.exists():
        logger.info(
            "Source metadata not found at %s — leaf grouping disabled.",
            source_metadata_path,
        )
        return {}

    df = pd.read_csv(source_metadata_path, encoding="utf-8")

    if "image_path" not in df.columns or "leaf_id" not in df.columns:
        logger.warning(
            "Source metadata missing required columns (image_path, leaf_id). "
            "Leaf grouping disabled."
        )
        return {}

    leaf_map: dict[str, str] = {}
    for _, row in df.iterrows():
        normalised = str(Path(str(row["image_path"])).resolve())
        leaf_map[normalised] = str(row["leaf_id"])

    logger.info(
        "Loaded source metadata: %d entries with leaf_id from %s",
        len(leaf_map),
        source_metadata_path,
    )
    return leaf_map


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------


def check_dataset_exists(
    raw_dir: Path, class_map: dict[str, str] = CLASS_MAP
) -> bool:
    """Check whether the raw dataset directory exists."""
    if not raw_dir.exists():
        logger.error("=" * 60)
        logger.error("Dataset not found at: %s", raw_dir)
        logger.error("")
        logger.error("To prepare the dataset:")
        logger.error("  1. Download the PlantVillage dataset.")
        logger.error("  2. Extract the archive.")
        logger.error("  3. Copy ONLY these three folders:")
        for class_name in class_map:
            logger.error("     - %s", class_name)
        logger.error("  4. Place them inside: %s", raw_dir)
        logger.error("")
        logger.error("Expected structure:")
        logger.error("  %s/", raw_dir)
        for class_name in class_map:
            logger.error("    ├── %s/", class_name)
        logger.error("=" * 60)
        return False
    return True


def check_class_dirs(
    raw_dir: Path, class_map: dict[str, str] = CLASS_MAP
) -> bool:
    """Verify that all expected class directories exist."""
    all_present = True
    for class_name in class_map:
        class_dir = raw_dir / class_name
        if class_dir.exists() and class_dir.is_dir():
            logger.info("  [OK] %s/", class_name)
        else:
            logger.error("  [MISSING] %s/", class_name)
            all_present = False
    return all_present


def scan_dataset(
    raw_dir: Path, class_map: dict[str, str] = CLASS_MAP
) -> tuple[list[dict], list[str]]:
    """
    Recursively scan the dataset directory.

    Returns:
        Tuple of (valid_records, corrupted_files).
    """
    records: list[dict] = []
    corrupted: list[str] = []

    expected_classes = list(class_map.keys())

    # Check for unexpected directories
    for item in sorted(raw_dir.iterdir()):
        if item.is_dir() and item.name not in expected_classes:
            logger.warning("Unexpected directory (will be skipped): %s", item)

    # Scan expected class directories
    for class_name in expected_classes:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            logger.error("Class directory missing: %s", class_dir)
            continue

        clean_label = class_map[class_name]
        warned_extensions: set[str] = set()
        file_count = 0

        for file_path in sorted(class_dir.rglob("*")):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()

            if ext not in VALID_EXTENSIONS:
                if ext not in warned_extensions:
                    logger.warning(
                        "Unsupported format '%s' in %s (skipping)", ext, class_name
                    )
                    warned_extensions.add(ext)
                continue

            file_count += 1
            info = validate_image(file_path)

            if info is not None:
                info["raw_class"] = class_name
                info["clean_class"] = clean_label
                records.append(info)
            else:
                corrupted.append(str(file_path))

        logger.info("  %s: %d valid images scanned", class_name, file_count)

    logger.info(
        "Scan complete: %d valid, %d corrupted, %d classes",
        len(records),
        len(corrupted),
        len(expected_classes),
    )
    return records, corrupted


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def find_duplicates(records: list[dict]) -> dict[str, list[str]]:
    """
    Identify exact duplicate files by MD5 hash.

    Returns a dict mapping each duplicate MD5 to the list of file paths
    sharing that hash.
    """
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        hash_groups[record["md5"]].append(record["image_path"])

    duplicates = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}

    if duplicates:
        total_dup_files = sum(len(p) - 1 for p in duplicates.values())
        logger.warning(
            "Found %d duplicate groups (%d extra files)",
            len(duplicates),
            total_dup_files,
        )
        for md5, paths in duplicates.items():
            logger.warning("  MD5 %s: %d files", md5[:12], len(paths))
    else:
        logger.info("No exact duplicates found.")

    return duplicates


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def generate_summary(
    records: list[dict], corrupted: list[str], duplicates: dict[str, list[str]]
) -> dict:
    """Generate a dataset summary dictionary."""
    df = pd.DataFrame(records)
    total = len(df)

    class_counts = df["clean_class"].value_counts().to_dict() if total > 0 else {}

    format_counts = (
        df["file_extension"].value_counts().to_dict() if total > 0 else {}
    )
    mode_counts = df["image_mode"].value_counts().to_dict() if total > 0 else {}

    total_dup_files = sum(len(p) - 1 for p in duplicates.values())

    summary = {
        "total_images": total,
        "class_counts": class_counts,
        "class_percentages": (
            {k: round(v / total * 100, 2) for k, v in class_counts.items()}
            if total > 0
            else {}
        ),
        "corrupted_count": len(corrupted),
        "corrupted_files": corrupted,
        "duplicate_groups": len(duplicates),
        "duplicate_extra_files": total_dup_files,
        "format_counts": format_counts,
        "mode_counts": mode_counts,
    }

    logger.info("Dataset summary:")
    logger.info("  Total valid images: %d", total)
    for cls, count in class_counts.items():
        pct = summary["class_percentages"].get(cls, 0)
        logger.info("    %s: %d (%.1f%%)", cls, count, pct)
    logger.info("  Corrupted files: %d", len(corrupted))
    logger.info("  Duplicate extra files: %d", total_dup_files)
    logger.info("  Formats: %s", format_counts)
    logger.info("  Modes: %s", mode_counts)

    return summary

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_metadata_csv(records: list[dict], output_path: Path) -> None:
    """Write per-image metadata to CSV."""
    columns = [
        "image_path",
        "raw_class",
        "clean_class",
        "file_extension",
        "image_mode",
        "width",
        "height",
        "md5",
    ]
    # Include leaf_id column when source metadata was merged
    if records and "leaf_id" in records[0]:
        columns.append("leaf_id")
    df = pd.DataFrame(records, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Metadata CSV written: %s (%d rows)", output_path, len(df))


def write_dataset_report(
    summary: dict,
    corrupted: list[str],
    duplicates: dict[str, list[str]],
    output_path: Path,
    class_map: dict[str, str] = CLASS_MAP,
) -> None:
    """Write a human-readable dataset report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    crop_name = next(iter(class_map)).split("___", 1)[0]

    lines = [
        "=" * 60,
        "AgriMind AI — Dataset Report",
        "=" * 60,
        "",
        f"Dataset: PlantVillage ({crop_name} Subset)",
        f"Generated: {timestamp}",
        "",
        "Expected Classes:",
    ]
    for raw, clean in class_map.items():
        lines.append(f"  - {raw} -> {clean}")

    lines.extend(["", "Class Counts:"])
    for cls, count in summary["class_counts"].items():
        pct = summary["class_percentages"].get(cls, 0)
        lines.append(f"  {cls}: {count} ({pct}%)")

    lines.extend(
        [
            "",
            f"Total valid images: {summary['total_images']}",
            f"Corrupted files: {summary['corrupted_count']}",
            f"Duplicate groups: {summary['duplicate_groups']}",
            f"Duplicate extra files: {summary['duplicate_extra_files']}",
        ]
    )

    if corrupted:
        lines.append("")
        lines.append("Corrupted Files:")
        for f in corrupted:
            lines.append(f"  - {f}")

    if duplicates:
        lines.append("")
        lines.append("Duplicate Files (by MD5):")
        for md5, paths in duplicates.items():
            lines.append(f"  MD5 {md5}:")
            for p in paths:
                lines.append(f"    - {p}")

    lines.extend(["", "Image Format Summary:"])
    for fmt, count in summary["format_counts"].items():
        lines.append(f"  {fmt}: {count}")

    lines.extend(["", "Image Mode Summary:"])
    for mode, count in summary["mode_counts"].items():
        lines.append(f"  {mode}: {count}")

    lines.extend(
        [
            "",
            "NOTE: Raw data was NOT modified by this script.",
            "NOTE: No files were deleted or moved.",
            "=" * 60,
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Dataset report written: %s", output_path)


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------


def generate_splits(
    records: list[dict],
    duplicates: dict[str, list[str]],
    seed: int,
    splits_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    leaf_id_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test split manifests.

    Duplicate handling: keeps only the first file per MD5 group
    so that identical images never appear in different splits.

    Leaf grouping: when *leaf_id_map* is provided (mapping image_path
    to leaf_id), all images from the same leaf stay in one split.
    This prevents data leakage from multiple photos of the same leaf.
    Falls back to standard stratified splitting when not available.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate: keep only first occurrence per MD5
    dedup_df = pd.DataFrame(records)

    if duplicates:
        before = len(dedup_df)
        dedup_df = dedup_df.drop_duplicates(subset="md5", keep="first")
        removed = before - len(dedup_df)
        logger.info(
            "Deduplication: kept first of each MD5 group (%d records removed)",
            removed,
        )
    else:
        logger.info("No duplicates to remove before splitting.")

    # Reset index for positional alignment
    dedup_df = dedup_df.reset_index(drop=True)

    # ---- Leaf-group-aware splitting ----
    # Build leaf groups: all images from the same leaf stay together.
    leaf_groups: dict[str, list[int]] = defaultdict(list)
    leaf_labels: dict[str, list[str]] = defaultdict(list)

    if leaf_id_map:
        _ungrouped_counter = 0
        for idx, row in dedup_df.iterrows():
            normalised = str(Path(row["image_path"]).resolve())
            lid = leaf_id_map.get(normalised)
            if lid is None:
                continue
            # Treat unverified leaf IDs as individual single-image groups
            # so they distribute evenly across splits instead of forming
            # one mega-group.
            if not is_verified_leaf_id(lid):
                lid = f"__ungrouped_{_ungrouped_counter}__"
                _ungrouped_counter += 1
            leaf_groups[lid].append(idx)
            leaf_labels[lid].append(row["clean_class"])

    if not leaf_groups:
        # Fallback: standard stratified split (no leaf grouping)
        logger.info("No leaf grouping available — using stratified split.")
        paths = dedup_df["image_path"].to_numpy()
        labels = dedup_df["clean_class"].to_numpy()

        test_val_ratio = val_ratio + test_ratio
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels, test_size=test_val_ratio,
            random_state=seed, stratify=labels,
        )

        val_fraction = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=(1 - val_fraction),
            random_state=seed, stratify=temp_labels,
        )

        def make_split_df(paths_arr, labels_arr):
            return pd.DataFrame(
                {"image_path": paths_arr, "clean_class": labels_arr}
            )

        train_df = make_split_df(train_paths, train_labels)
        val_df = make_split_df(val_paths, val_labels)
        test_df = make_split_df(test_paths, test_labels)

    else:
        # Leaf-group-aware split: keep all images of one leaf in one split.
        # Use majority class per leaf group for stratification.
        logger.info(
            "Leaf grouping enabled: %d unique leaves across %d images.",
            len(leaf_groups),
            sum(len(v) for v in leaf_groups.values()),
        )

        leaf_ids = []
        majority_labels = []
        group_indices = []

        for lid, indices in leaf_groups.items():
            leaf_ids.append(lid)
            group_indices.append(indices)
            label_counts = pd.Series(leaf_labels[lid]).value_counts()
            majority_labels.append(label_counts.index[0])

        leaf_ids_arr = np.array(leaf_ids)
        majority_arr = np.array(majority_labels)
        idx_list = list(range(len(leaf_ids)))
        indices_arr = np.array(idx_list)

        test_val_ratio = val_ratio + test_ratio
        train_lidx, temp_lidx, train_lab, temp_lab = train_test_split(
            indices_arr, majority_arr, test_size=test_val_ratio,
            random_state=seed, stratify=majority_arr,
        )

        val_fraction = val_ratio / (val_ratio + test_ratio)
        val_lidx, test_lidx, val_lab, test_lab = train_test_split(
            temp_lidx, temp_lab, test_size=(1 - val_fraction),
            random_state=seed, stratify=temp_lab,
        )

        train_indices = []
        for li in train_lidx:
            train_indices.extend(group_indices[li])
        val_indices = []
        for li in val_lidx:
            val_indices.extend(group_indices[li])
        test_indices = []
        for li in test_lidx:
            test_indices.extend(group_indices[li])

        train_df = pd.DataFrame({
            "image_path": dedup_df["image_path"].iloc[train_indices].to_numpy(),
            "clean_class": dedup_df["clean_class"].iloc[train_indices].to_numpy(),
        })
        val_df = pd.DataFrame({
            "image_path": dedup_df["image_path"].iloc[val_indices].to_numpy(),
            "clean_class": dedup_df["clean_class"].iloc[val_indices].to_numpy(),
        })
        test_df = pd.DataFrame({
            "image_path": dedup_df["image_path"].iloc[test_indices].to_numpy(),
            "clean_class": dedup_df["clean_class"].iloc[test_indices].to_numpy(),
        })

    # Write CSVs
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = splits_dir / f"{name}.csv"
        df.to_csv(out, index=False, encoding="utf-8")
        logger.info("  %s split: %d images -> %s", name, len(df), out)

    return train_df, val_df, test_df


def write_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path,
    leaf_id_map: dict[str, str] | None = None,
) -> None:
    """Write a verification report for the train/val/test splits."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(train_df) + len(val_df) + len(test_df)

    def class_breakdown(df: pd.DataFrame) -> dict[str, int]:
        return df["clean_class"].value_counts().to_dict()

    train_counts = class_breakdown(train_df)
    val_counts = class_breakdown(val_df)
    test_counts = class_breakdown(test_df)

    # Leakage check: no path should appear in more than one split
    train_set = set(train_df["image_path"])
    val_set = set(val_df["image_path"])
    test_set = set(test_df["image_path"])

    tv_overlap = train_set & val_set
    tt_overlap = train_set & test_set
    vt_overlap = val_set & test_set
    no_overlap = (
        len(tv_overlap) == 0 and len(tt_overlap) == 0 and len(vt_overlap) == 0
    )

    def pct(n: int) -> str:
        return f"{n / total * 100:.1f}%" if total > 0 else "0.0%"

    lines = [
        "=" * 60,
        "AgriMind AI — Split Report",
        "=" * 60,
        "",
        f"Generated: {timestamp}",
        "",
        "Split Sizes:",
        f"  Train: {len(train_df)} ({pct(len(train_df))})",
        f"  Validation: {len(val_df)} ({pct(len(val_df))})",
        f"  Test: {len(test_df)} ({pct(len(test_df))})",
        f"  Total: {total}",
        "",
        "Per-Class Counts:",
        "",
        "  Train:",
    ]
    for cls, count in sorted(train_counts.items()):
        lines.append(f"    {cls}: {count}")

    lines.append("")
    lines.append("  Validation:")
    for cls, count in sorted(val_counts.items()):
        lines.append(f"    {cls}: {count}")

    lines.append("")
    lines.append("  Test:")
    for cls, count in sorted(test_counts.items()):
        lines.append(f"    {cls}: {count}")

    lines.extend(
        [
            "",
            "Data Leakage Check:",
            f"  Train-Val overlap: {len(tv_overlap)}",
            f"  Train-Test overlap: {len(tt_overlap)}",
            f"  Val-Test overlap: {len(vt_overlap)}",
            f"  RESULT: {'PASS - No overlap detected' if no_overlap else 'FAIL - Overlap detected!'}",
        ]
    )

    # ---- Leaf-level leakage check ----
    # Use the SAME effective grouping as generate_splits(): verified leaves
    # keep their real ID; unverified images ("unknown", "fallback_*") are
    # each treated as an individual single-image group.
    if leaf_id_map:
        verified_leaf_map: dict[str, str] = {}  # path -> leaf_id
        for p_raw in train_df["image_path"].tolist() + val_df["image_path"].tolist() + test_df["image_path"].tolist():
            norm = str(Path(p_raw).resolve())
            lid = leaf_id_map.get(norm)
            if lid is None:
                continue
            if is_verified_leaf_id(lid):
                verified_leaf_map[norm] = lid

        # Build verified-leaf sets per split
        def _verified_leaves_for(df: pd.DataFrame) -> set[str]:
            return {
                verified_leaf_map[str(Path(p).resolve())]
                for p in df["image_path"]
                if str(Path(p).resolve()) in verified_leaf_map
            }

        train_vl = _verified_leaves_for(train_df)
        val_vl = _verified_leaves_for(val_df)
        test_vl = _verified_leaves_for(test_df)
        all_verified = train_vl | val_vl | test_vl

        lv_tv = train_vl & val_vl
        lv_tt = train_vl & test_vl
        lv_vt = val_vl & test_vl
        no_leaf_overlap = (
            len(lv_tv) == 0 and len(lv_tt) == 0 and len(lv_vt) == 0
        )

        # Count ungrouped images per split
        def _ungrouped_for(df: pd.DataFrame) -> int:
            return sum(
                1 for p in df["image_path"]
                if str(Path(p).resolve()) not in verified_leaf_map
                and leaf_id_map.get(str(Path(p).resolve())) is not None
            )

        train_ungrouped = _ungrouped_for(train_df)
        val_ungrouped = _ungrouped_for(val_df)
        test_ungrouped = _ungrouped_for(test_df)
        total_ungrouped = train_ungrouped + val_ungrouped + test_ungrouped
        effective_groups = len(all_verified) + total_ungrouped

        # Overall leakage = image overlap OR verified-leaf overlap
        overall_pass = no_overlap and no_leaf_overlap

        lines.extend(
            [
                "",
                "Leaf Grouping:",
                f"  Verified leaves: {len(all_verified)}",
                f"  Ungrouped (unverified) images: {total_ungrouped}",
                f"  Effective groups: {effective_groups}",
                f"  Train: {len(train_vl)} verified leaves, {train_ungrouped} ungrouped",
                f"  Validation: {len(val_vl)} verified leaves, {val_ungrouped} ungrouped",
                f"  Test: {len(test_vl)} verified leaves, {test_ungrouped} ungrouped",
                "",
                "  Verified leaf overlap:",
                f"    Train-Val: {len(lv_tv)}",
                f"    Train-Test: {len(lv_tt)}",
                f"    Val-Test: {len(lv_vt)}",
                f"    RESULT: {'PASS' if no_leaf_overlap else 'FAIL - Verified leaf overlap detected!'}",
                "",
                f"  Overall leakage: {'PASS' if overall_pass else 'FAIL'}",
            ]
        )

    notes = [
        "",
        "NOTE: Exact duplicates (by MD5) were removed before splitting.",
        "NOTE: Splits are stratified to preserve class proportions.",
    ]
    if leaf_id_map:
        notes.append(
            "NOTE: Verified leaf grouping was used — all images from the same "
            "physical leaf are in a single split. Unverified images are each "
            "treated as an individual group."
        )
    notes.append("=" * 60)

    lines.extend(notes)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Split report written: %s", output_path)

    if not no_overlap:
        logger.error("DATA LEAKAGE DETECTED - check split report immediately!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full dataset preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="AgriMind AI - Dataset Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.prepare_dataset\n"
            "  python -m src.prepare_dataset --skip-split\n"
            "  python -m src.prepare_dataset --config configs/config.yaml\n"
            "  python -m src.prepare_dataset "
            "--config configs/config_potato.yaml\n"
            "\n"
            "Prerequisites:\n"
            "  Tomato: place the PlantVillage tomato subset in:\n"
            "    data/raw/tomato/Tomato___healthy/\n"
            "    data/raw/tomato/Tomato___Early_blight/\n"
            "    data/raw/tomato/Tomato___Late_blight/\n"
            "  Potato: place the PlantVillage potato subset in:\n"
            "    data/raw/potato/Potato___healthy/\n"
            "    data/raw/potato/Potato___Early_blight/\n"
            "    data/raw/potato/Potato___Late_blight/\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Only scan and verify - skip train/val/test split generation",
    )
    parser.add_argument(
        "--source-metadata",
        type=str,
        default=None,
        help=(
            "Path to source metadata CSV with leaf_id column "
            "(enables leaf-group-aware splitting)"
        ),
    )
    args = parser.parse_args()

    # ---- Load configuration ----
    config = load_config(args.config)
    seed = config.get("seed", 42)
    data_paths = config.get("data", {})
    crop = config.get("crop", "tomato")
    class_map = CROP_CLASS_MAPS.get(crop, CLASS_MAP)
    raw_dir = Path(data_paths.get("raw", "data/raw")) / crop
    processed_dir = Path(data_paths.get("processed", "data/processed"))
    splits_dir = Path(data_paths.get("splits", "data/splits"))

    logger.info("=" * 60)
    logger.info("AgriMind AI - Dataset Preparation Pipeline")
    logger.info("Crop: %s", crop.capitalize())
    logger.info("=" * 60)

    # ---- Ensure output directories exist ----
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load optional source metadata (leaf grouping) ----
    if args.source_metadata:
        leaf_id_map = load_source_metadata(Path(args.source_metadata))
        logger.info(
            "Leaf grouping: %s (%d entries)",
            "enabled" if leaf_id_map else "no matching entries",
            len(leaf_id_map),
        )
    else:
        leaf_id_map: dict[str, str] = {}
        logger.info("Leaf grouping: not requested (use --source-metadata to enable)")

    # ---- Step 1: Verify dataset exists ----
    logger.info("Step 1: Checking dataset location...")
    if not check_dataset_exists(raw_dir, class_map):
        sys.exit(1)

    # ---- Step 2: Verify class directories ----
    logger.info("Step 2: Verifying class directories...")
    if not check_class_dirs(raw_dir, class_map):
        logger.error("One or more expected class directories are missing.")
        sys.exit(1)

    # ---- Step 3: Scan and validate ----
    logger.info("Step 3: Scanning and validating images...")
    records, corrupted = scan_dataset(raw_dir, class_map)

    if not records:
        logger.error("No valid images found. Check your dataset and try again.")
        sys.exit(1)

    # ---- Merge leaf_id from source metadata into records ----
    if leaf_id_map:
        for record in records:
            normalised = str(Path(record["image_path"]).resolve())
            record["leaf_id"] = leaf_id_map.get(normalised)
        matched = sum(1 for r in records if r.get("leaf_id") is not None)
        logger.info(
            "Merged leaf_id into %d / %d records.", matched, len(records)
        )

    # ---- Step 4: Detect duplicates ----
    logger.info("Step 4: Detecting exact duplicates (MD5)...")
    duplicates = find_duplicates(records)

    # ---- Step 5: Generate summary ----
    logger.info("Step 5: Generating dataset summary...")
    summary = generate_summary(records, corrupted, duplicates)

    # ---- Step 6: Write metadata CSV ----
    logger.info("Step 6: Writing metadata CSV...")
    write_metadata_csv(records, processed_dir / "dataset_metadata.csv")

    # ---- Step 7: Write dataset report ----
    logger.info("Step 7: Writing dataset report...")
    write_dataset_report(
        summary, corrupted, duplicates, processed_dir / "dataset_report.txt",
        class_map,
    )

    # ---- Step 8: Generate splits ----
    if args.skip_split:
        logger.info("Step 8: Skipped (--skip-split flag set).")
    else:
        logger.info("Step 8: Generating stratified splits (70/15/15)...")
        train_df, val_df, test_df = generate_splits(
            records, duplicates, seed, splits_dir,
            leaf_id_map=leaf_id_map,
        )

        logger.info("Step 9: Writing split report...")
        write_split_report(
            train_df, val_df, test_df, splits_dir / "split_report.txt",
            leaf_id_map=leaf_id_map,
        )

    logger.info("=" * 60)
    logger.info("Pipeline complete. Raw data was NOT modified.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
