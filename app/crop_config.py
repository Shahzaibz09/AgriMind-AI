"""
AgriMind AI — Crop Configuration Registry.

Discovers and loads crop-specific YAML configs from ``configs/crops/``.
Each crop has its own file (e.g., ``tomato.yaml``, ``potato.yaml``) that
defines the model path, class names, image size, and display metadata.

This module has no Streamlit dependency — it is pure Python and can be
unit-tested independently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default directory for crop configs (relative to project root)
DEFAULT_CROPS_DIR = "configs/crops"

# Required keys in every crop config
_REQUIRED_KEYS = {"crop_name", "model_path", "num_classes", "class_names", "image_size"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_crops(crops_dir: str | Path = DEFAULT_CROPS_DIR) -> list[str]:
    """Return sorted list of available crop names.

    Scans ``crops_dir`` for ``*.yaml`` files and returns the crop_name
    field from each.

    Parameters
    ----------
    crops_dir : str or Path
        Directory containing crop YAML files.

    Returns
    -------
    list of crop name strings.
    """
    crops_dir = Path(crops_dir)
    if not crops_dir.is_dir():
        return []

    names: list[str] = []
    for yaml_file in sorted(crops_dir.glob("*.yaml")):
        try:
            cfg = _load_yaml(yaml_file)
            name = cfg.get("crop_name")
            if name:
                names.append(name)
        except Exception:
            logger.warning("Skipping invalid crop config: %s", yaml_file)

    return sorted(names)


def get_crop_config(
    crop_name: str,
    crops_dir: str | Path = DEFAULT_CROPS_DIR,
) -> dict[str, Any]:
    """Load and validate the config for a specific crop.

    Parameters
    ----------
    crop_name : str
        Crop name (must match ``crop_name`` field in a YAML file).
    crops_dir : str or Path
        Directory containing crop YAML files.

    Returns
    -------
    dict with all crop configuration fields.

    Raises
    ------
    FileNotFoundError
        If no matching crop config is found.
    ValueError
        If the config is missing required keys.
    """
    crops_dir = Path(crops_dir)
    if not crops_dir.is_dir():
        raise FileNotFoundError(f"Crops directory not found: {crops_dir}")

    for yaml_file in sorted(crops_dir.glob("*.yaml")):
        cfg = _load_yaml(yaml_file)
        if cfg.get("crop_name") == crop_name:
            _validate(crop_name, cfg)
            # Attach the source file path for debugging
            cfg["_config_file"] = str(yaml_file)
            return cfg

    available = list_crops(crops_dir)
    raise FileNotFoundError(
        f"Crop '{crop_name}' not found. Available: {available}"
    )


def load_all_crops(
    crops_dir: str | Path = DEFAULT_CROPS_DIR,
) -> dict[str, dict[str, Any]]:
    """Load all crop configs into a dict keyed by crop name.

    Parameters
    ----------
    crops_dir : str or Path
        Directory containing crop YAML files.

    Returns
    -------
    dict mapping crop_name -> config dict.
    """
    crops_dir = Path(crops_dir)
    if not crops_dir.is_dir():
        return {}

    result: dict[str, dict[str, Any]] = {}
    for yaml_file in sorted(crops_dir.glob("*.yaml")):
        try:
            cfg = _load_yaml(yaml_file)
            name = cfg.get("crop_name")
            if name:
                _validate(name, cfg)
                cfg["_config_file"] = str(yaml_file)
                result[name] = cfg
        except Exception:
            logger.warning("Skipping invalid crop config: %s", yaml_file)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def _validate(crop_name: str, cfg: dict[str, Any]) -> None:
    """Validate that all required keys are present."""
    missing = _REQUIRED_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(
            f"Crop '{crop_name}' config missing required keys: {missing}"
        )
