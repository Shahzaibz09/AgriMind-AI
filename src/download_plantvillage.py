"""
AgriMind AI — PlantVillage Dataset Acquisition Script (Direct Download)

Downloads the tomato or potato subset of the PlantVillage dataset from
Hugging Face (mohanty/PlantVillage) using direct file acquisition via
huggingface_hub, then selectively extracts target images from the
data.zip archive.

Prerequisites:
    pip install huggingface_hub Pillow pandas

Usage:
    python -m src.download_plantvillage
    python -m src.download_plantvillage --help
    python -m src.download_plantvillage --force
    python -m src.download_plantvillage --output-dir data/raw/tomato
    python -m src.download_plantvillage --crop potato
"""

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src.prepare_dataset import CLASS_MAP, CROP_CLASS_MAPS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_ID = "mohanty/PlantVillage"

# Default (tomato) target classes. Potato targets are resolved at runtime
# via CROP_CLASS_MAPS — see download_pipeline(crop=...).
TARGET_CLASSES = list(CLASS_MAP.keys())
TARGET_CLASS_SET = frozenset(TARGET_CLASSES)

REQUIRED_FILES = [
    "data.zip",
    "leaf_grouping/leaf-map.json",
    "splits/color_train.txt",
    "splits/color_test.txt",
]

IMAGE_QUALITY = 95  # JPEG quality when converting non-JPEG images

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
# Download helpers
# ---------------------------------------------------------------------------


def download_repo_file(filename: str, repo_id: str = HF_REPO_ID) -> Path:
    """Download a single file from the HF repository.

    Returns the local path to the cached file.
    """
    from huggingface_hub import hf_hub_download

    logger.info("  Downloading: %s", filename)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )
        p = Path(local_path)
        size_mb = p.stat().st_size / (1024 * 1024)
        logger.info("    Cached: %s (%.1f MB)", p, size_mb)
        return p
    except Exception as e:
        logger.error("Failed to download %s: %s", filename, e)
        raise


def get_file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# Split-file parsing
# ---------------------------------------------------------------------------


def parse_split_file(split_path: Path) -> list[str]:
    """Parse a color split file and return non-empty stripped paths.

    Raises ``ValueError`` on malformed lines.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    paths: list[str] = []
    with open(split_path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            if "/" not in line:
                raise ValueError(
                    f"Malformed line {line_no} in {split_path}: {line!r}"
                )
            paths.append(line)
    return paths


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def extract_class_from_path(source_path: str) -> str:
    """Extract the class directory name from a PlantVillage source path.

    Example::

        raw/color/Tomato___healthy/uuid.JPG  ->  Tomato___healthy
    """
    parts = source_path.replace("\\", "/").split("/")
    if len(parts) < 3:
        return ""
    return parts[2]


def filter_target_paths(
    paths: list[str],
    target_classes: set[str] | frozenset[str],
) -> list[str]:
    """Return only paths whose class directory is in *target_classes*."""
    return [p for p in paths if extract_class_from_path(p) in target_classes]


def combine_splits(
    train_paths: list[str],
    test_paths: list[str],
) -> list[dict]:
    """Combine train and test paths, tagging each with its source_split.

    Duplicate paths are silently dropped (first occurrence wins).
    """
    seen: set[str] = set()
    records: list[dict] = []
    for path, split_name in (
        *[(p, "train") for p in train_paths],
        *[(p, "test") for p in test_paths],
    ):
        if path in seen:
            continue
        seen.add(path)
        records.append({
            "source_path": path,
            "source_split": split_name,
            "class_name": extract_class_from_path(path),
        })
    return records


# ---------------------------------------------------------------------------
# Leaf-map resolution
# ---------------------------------------------------------------------------


def load_leaf_map(leaf_map_path: Path) -> dict[str, list[str]]:
    """Load the leaf-map.json file.

    Returns the raw dict: ``{short_id_lowercase: ["class:::leaf_num", ...]}``.
    """
    if not leaf_map_path.exists():
        raise FileNotFoundError(f"Leaf-map not found: {leaf_map_path}")
    with open(leaf_map_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"leaf-map.json root must be a dict, got {type(data).__name__}"
        )
    return data


def extract_short_id(source_path: str) -> str:
    """Extract the short identifier from a PlantVillage image filename.

    Given ``raw/color/Tomato___healthy/UUID___ShortID.JPG``,
    returns ``ShortID`` (without extension).
    """
    filename = source_path.replace("\\", "/").split("/")[-1]
    ext_idx = filename.rfind(".")
    if ext_idx > 0:
        name = filename[:ext_idx]
    else:
        name = filename
    sep = "___"
    idx = name.rfind(sep)
    if idx < 0:
        return name
    return name[idx + len(sep):]


def resolve_leaf_id(
    short_id: str,
    leaf_map: dict[str, list[str]],
) -> tuple[str, str]:
    """Resolve a leaf_id from the leaf-map.

    Returns ``(leaf_id, status)`` where *status* is one of:
    ``"verified"``, ``"fallback"``, ``"unknown"``.
    """
    lookup = short_id.strip().lower()

    entries = leaf_map.get(lookup)
    if not entries:
        return ("unknown", "unknown")

    raw_value = entries[0]  # first mapping
    if not raw_value or not isinstance(raw_value, str):
        return ("unknown", "unknown")

    # Detect fallback markers in the value
    lower_val = raw_value.lower()
    if "fallback" in lower_val or lower_val.startswith("unknown"):
        return (raw_value, "fallback")

    # Parse "ClassName:::LeafNum" -> construct leaf_id
    if ":::" in raw_value:
        class_name, leaf_num_str = raw_value.split(":::", 1)
        try:
            leaf_num = int(float(leaf_num_str))
        except (ValueError, TypeError):
            return (raw_value, "verified")
        return (f"{class_name}:::{leaf_num}", "verified")

    return (raw_value, "verified")


def resolve_all_leaf_ids(
    records: list[dict],
    leaf_map: dict[str, list[str]],
) -> list[dict]:
    """Add leaf_id and leaf_id_status to each record.

    Mutates and returns the same list.
    """
    verified = fallback = unknown = 0
    for record in records:
        short_id = extract_short_id(record["source_path"])
        leaf_id, status = resolve_leaf_id(short_id, leaf_map)
        record["leaf_id"] = leaf_id
        record["leaf_id_status"] = status
        if status == "verified":
            verified += 1
        elif status == "fallback":
            fallback += 1
        else:
            unknown += 1

    logger.info(
        "Leaf-ID resolution: %d verified, %d fallback, %d unknown",
        verified,
        fallback,
        unknown,
    )
    return records


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------


def _normalise_zip_member(name: str) -> str:
    """Normalise a ZIP member name to forward-slash, stripped."""
    return name.replace("\\", "/").strip("/")


def _build_member_index(
    zf: zipfile.ZipFile,
) -> dict[str, str]:
    """Build a mapping ``{normalised_name: original_name}`` for ZIP members."""
    index: dict[str, str] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        norm = _normalise_zip_member(info.filename)
        index[norm] = info.filename
    return index


def extract_target_images(
    zip_path: Path,
    records: list[dict],
    output_dir: Path,
    skip_existing: bool = True,
    class_map: dict[str, str] = CLASS_MAP,
    crop_label: str = "Tomato",
) -> list[dict]:
    """Selectively extract target images from data.zip.

    Returns a list of metadata dicts for each extracted (or skipped) image.
    *class_map* maps raw class folder names to clean class names and
    *crop_label* is recorded in the metadata (both default to tomato).
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"data.zip not found: {zip_path}")

    try:
        zf = zipfile.ZipFile(str(zip_path), "r")
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Corrupted ZIP archive: {zip_path}") from exc

    member_index = _build_member_index(zf)

    saved: list[dict] = []
    extracted = 0
    skipped = 0
    missing = 0

    for record in records:
        source_path = record["source_path"]
        class_name = record["class_name"]
        norm_path = _normalise_zip_member(source_path)

        # Locate the member inside the ZIP
        zip_member = member_index.get(norm_path)
        if zip_member is None:
            logger.warning(
                "  ZIP member not found: %s", source_path,
            )
            missing += 1
            continue

        # Determine output location
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / Path(source_path).name

        if skip_existing and out_path.exists():
            skipped += 1
            saved.append(
                _build_saved_record(record, out_path, class_map, crop_label)
            )
            continue

        # Extract: write raw bytes (no re-encoding)
        try:
            data = zf.read(zip_member)
            out_path.write_bytes(data)
            extracted += 1
        except Exception as exc:
            logger.error("Failed to extract %s: %s", source_path, exc)
            missing += 1
            continue

        saved.append(
            _build_saved_record(record, out_path, class_map, crop_label)
        )

    zf.close()

    logger.info(
        "Extraction: %d extracted, %d skipped (existing), %d missing in ZIP",
        extracted,
        skipped,
        missing,
    )
    return saved


def _build_saved_record(
    record: dict,
    out_path: Path,
    class_map: dict[str, str] = CLASS_MAP,
    crop_label: str = "Tomato",
) -> dict:
    """Build a metadata dict for a saved/skipped image."""
    return {
        "image_path": str(out_path.resolve()),
        "leaf_id": record["leaf_id"],
        "original_label": record["class_name"],
        "clean_class": class_map[record["class_name"]],
        "source_split": record["source_split"],
        "crop": crop_label,
        "disease": record["class_name"].split("___")[-1],
        "leaf_id_status": record["leaf_id_status"],
    }


# ---------------------------------------------------------------------------
# Metadata CSV
# ---------------------------------------------------------------------------

_METADATA_COLUMNS = [
    "image_path",
    "leaf_id",
    "original_label",
    "clean_class",
    "source_split",
    "crop",
    "disease",
    "leaf_id_status",
]


def write_source_metadata_csv(
    saved_records: list[dict],
    output_path: Path,
) -> None:
    """Write source metadata CSV with all required columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(saved_records, columns=_METADATA_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(
        "Source metadata CSV written: %s (%d rows)", output_path, len(df),
    )


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------


def check_output_exists(
    output_dir: Path,
    target_classes: list[str],
) -> bool:
    """Check whether any target class directory already contains images."""
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for class_name in target_classes:
        class_dir = output_dir / class_name
        if class_dir.exists() and any(
            f.is_file() and f.suffix.lower() in valid_ext
            for f in class_dir.iterdir()
        ):
            return True
    return False


def cleanup_generated_output(
    output_dir: Path,
    metadata_path: Path,
    target_classes: list[str],
) -> None:
    """Remove previously generated images and metadata."""
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for class_name in target_classes:
        class_dir = output_dir / class_name
        if class_dir.exists():
            for f in class_dir.iterdir():
                if f.is_file() and f.suffix.lower() in valid_ext:
                    f.unlink()
            logger.info("  Cleaned: %s", class_dir)
    if metadata_path.exists():
        metadata_path.unlink()
        logger.info("  Removed: %s", metadata_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def download_pipeline(
    output_dir: Path,
    metadata_path: Path,
    skip_existing: bool = True,
    crop: str = "tomato",
) -> None:
    """Execute the full download -> extract -> metadata pipeline."""
    class_map = CROP_CLASS_MAPS.get(crop)
    if class_map is None:
        raise ValueError(
            f"Unknown crop {crop!r} — expected one of "
            f"{sorted(CROP_CLASS_MAPS)}"
        )
    target_classes = list(class_map.keys())
    target_class_set = frozenset(target_classes)
    crop_label = crop.capitalize()

    logger.info("Dataset source:  %s", HF_REPO_ID)
    logger.info("Crop:            %s", crop_label)
    logger.info("Target classes:  %s", target_classes)
    logger.info("Output:          %s", output_dir)
    logger.info("Metadata:        %s", metadata_path)

    # ---- 1. Download repository files ----
    logger.info("=" * 60)
    logger.info("Step 1/4: Downloading repository files")
    logger.info(
        "  NOTE: data.zip is ~2.03 GB. First run will download to HF cache."
    )
    logger.info("=" * 60)

    zip_path = download_repo_file("data.zip")
    leaf_map_path = download_repo_file("leaf_grouping/leaf-map.json")
    train_split_path = download_repo_file("splits/color_train.txt")
    test_split_path = download_repo_file("splits/color_test.txt")

    archive_mb = get_file_size_mb(zip_path)
    logger.info("Archive size: %.1f MB (%.2f GB)", archive_mb, archive_mb / 1024)

    # ---- 2. Parse split files ----
    logger.info("=" * 60)
    logger.info("Step 2/4: Parsing split files and filtering")
    logger.info("=" * 60)

    train_paths = parse_split_file(train_split_path)
    test_paths = parse_split_file(test_split_path)
    logger.info("Source train paths: %d", len(train_paths))
    logger.info("Source test paths:  %d", len(test_paths))
    logger.info("Total color paths: %d", len(train_paths) + len(test_paths))

    # Detect duplicates across splits
    overlap = set(train_paths) & set(test_paths)
    if overlap:
        logger.warning("Duplicate paths across splits: %d", len(overlap))

    # Filter target classes
    train_target = filter_target_paths(train_paths, target_class_set)
    test_target = filter_target_paths(test_paths, target_class_set)
    logger.info("Target train paths: %d", len(train_target))
    logger.info("Target test paths:  %d", len(test_target))

    # Combine and tag with source_split
    records = combine_splits(train_target, test_target)
    logger.info("Combined target records: %d", len(records))

    # Per-class counts
    class_counts: dict[str, int] = {}
    for r in records:
        cls = r["class_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    for target in target_classes:
        count = class_counts.get(target, 0)
        logger.info("  %s: %d images", target, count)
        if count == 0:
            logger.error(
                "Target class '%s' had zero matching records.", target,
            )
            raise SystemExit(1)

    # ---- 3. Resolve leaf IDs ----
    logger.info("=" * 60)
    logger.info("Step 3/4: Resolving leaf IDs")
    logger.info("=" * 60)

    leaf_map = load_leaf_map(leaf_map_path)
    logger.info("Leaf-map entries: %d", len(leaf_map))
    records = resolve_all_leaf_ids(records, leaf_map)

    # ---- 4. Extract images from ZIP ----
    logger.info("=" * 60)
    logger.info("Step 4/4: Extracting target images from archive")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = extract_target_images(
        zip_path, records, output_dir, skip_existing=skip_existing,
        class_map=class_map, crop_label=crop_label,
    )

    # ---- Write metadata CSV ----
    write_source_metadata_csv(saved, metadata_path)

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Acquisition complete.")
    logger.info("  Archive size:        %.2f GB", archive_mb / 1024)
    logger.info("  Source train paths:  %d", len(train_paths))
    logger.info("  Source test paths:   %d", len(test_paths))
    logger.info("  Total color paths:   %d", len(train_paths) + len(test_paths))
    for target in target_classes:
        logger.info("  %s: %d", target, class_counts.get(target, 0))
    logger.info("  Total selected:      %d", len(records))
    statuses: dict[str, int] = {}
    for r in saved:
        s = r.get("leaf_id_status", "")
        statuses[s] = statuses.get(s, 0) + 1
    logger.info("  Verified leaf IDs:   %d", statuses.get("verified", 0))
    logger.info("  Fallback leaf IDs:   %d", statuses.get("fallback", 0))
    logger.info("  Unknown leaf IDs:    %d", statuses.get("unknown", 0))
    logger.info("  Extracted images:    %d", len(saved))
    logger.info("  Output directory:    %s", output_dir)
    logger.info("  Metadata CSV:        %s", metadata_path)
    logger.info("")
    logger.info("Next step:")
    logger.info(
        "  python -m src.prepare_dataset --source-metadata %s",
        metadata_path,
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the PlantVillage dataset acquisition pipeline."""
    parser = argparse.ArgumentParser(
        description="AgriMind AI -- PlantVillage Dataset Acquisition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.download_plantvillage\n"
            "  python -m src.download_plantvillage --force\n"
            "  python -m src.download_plantvillage "
            "--output-dir data/raw/tomato\n"
            "  python -m src.download_plantvillage --crop potato\n"
            "\n"
            "Prerequisites:\n"
            "  pip install huggingface_hub Pillow pandas\n"
        ),
    )
    parser.add_argument(
        "--crop",
        type=str,
        choices=list(CROP_CLASS_MAPS),
        default="tomato",
        help="Crop subset to acquire (default: tomato)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for images (default: data/raw/<crop>)",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Output path for source metadata CSV "
        "(default: data/processed/source_metadata.csv for tomato, "
        "data/splits_<crop>/source_metadata.csv otherwise)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated dataset "
        "(only removes generated files, never unrelated files)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"data/raw/{args.crop}")
    default_metadata = (
        "data/processed/source_metadata.csv"
        if args.crop == "tomato"
        else f"data/splits_{args.crop}/source_metadata.csv"
    )
    metadata_path = Path(args.metadata_path or default_metadata)
    crop_target_classes = list(CROP_CLASS_MAPS[args.crop].keys())

    logger.info("=" * 60)
    logger.info("AgriMind AI -- PlantVillage Dataset Acquisition")
    logger.info("Crop:            %s", args.crop.capitalize())
    logger.info("=" * 60)

    # ---- Handle existing output ----
    if check_output_exists(output_dir, crop_target_classes):
        if args.force:
            logger.info(
                "--force: cleaning previously generated output in %s",
                output_dir,
            )
            cleanup_generated_output(
                output_dir, metadata_path, crop_target_classes,
            )
        else:
            logger.info(
                "Output already exists at %s. "
                "Existing images will be reused (use --force to re-extract).",
                output_dir,
            )

    try:
        download_pipeline(
            output_dir, metadata_path, skip_existing=True, crop=args.crop,
        )
    except SystemExit:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
