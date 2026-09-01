"""
AgriMind AI — MobileNetV3 Transfer Learning Model.

Provides a configurable MobileNetV3-Small model with a replaced
classifier head for tomato leaf disease classification.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHITECTURE_NAME = "mobilenet_v3_small"
DEFAULT_NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(
    num_classes: int = DEFAULT_NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a MobileNetV3-Small model for transfer learning.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 3).
    pretrained : bool
        If True, load ImageNet pretrained weights.
    freeze_backbone : bool
        If True, freeze all backbone parameters (only train classifier).

    Returns
    -------
    nn.Module with attributes:
        - ``num_classes``: int
        - ``architecture``: str
    """
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.mobilenet_v3_small(weights=weights)

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen (%d parameters)", _count_params(model.features))

    # Replace classifier head
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    # Attach metadata
    model.num_classes = num_classes  # type: ignore[attr-defined]
    model.architecture = ARCHITECTURE_NAME  # type: ignore[attr-defined]

    logger.info(
        "Model built: %s, num_classes=%d, trainable=%d/%d",
        ARCHITECTURE_NAME,
        num_classes,
        _count_trainable(model),
        _count_params(model),
    )
    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all backbone parameters for fine-tuning.

    Parameters
    ----------
    model : nn.Module
        Model previously created by :func:`build_model`.
    """
    for param in model.features.parameters():
        param.requires_grad = True
    logger.info(
        "Backbone unfrozen — trainable params: %d", _count_trainable(model)
    )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model state dict and metadata to a checkpoint file.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    path : str
        Output file path.
    metadata : dict, optional
        Additional metadata (hyperparameters, class mapping, etc.).
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": getattr(model, "architecture", ARCHITECTURE_NAME),
        "num_classes": getattr(model, "num_classes", DEFAULT_NUM_CLASSES),
    }
    if metadata:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, path)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(
    path: str,
    num_classes: int = DEFAULT_NUM_CLASSES,
    pretrained: bool = False,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load a model from a checkpoint file.

    Parameters
    ----------
    path : str
        Checkpoint file path.
    num_classes : int
        Number of output classes.
    pretrained : bool
        Whether to use pretrained backbone weights.
    device : str or torch.device
        Device to load model onto.

    Returns
    -------
    nn.Module with loaded weights.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        num_classes=checkpoint.get("num_classes", num_classes),
        pretrained=pretrained,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Checkpoint loaded: %s", path)
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_params(module: nn.Module) -> int:
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def _count_trainable(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return parameter counts for a model.

    Returns
    -------
    dict with 'total', 'trainable', 'frozen' keys.
    """
    total = _count_params(model)
    trainable = _count_trainable(model)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
