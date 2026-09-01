"""
AgriMind AI — Grad-CAM Explainability Module.

Hook-based Grad-CAM implementation targeting the last convolutional layer
of MobileNetV3-Small (``model.features[-1]``). Produces a class-discriminative
heatmap and an overlay visualisation.

No model architecture or checkpoint changes are required — Grad-CAM is
implemented purely via forward/backward hooks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, IDX_TO_CLASS, NUM_CLASSES, get_eval_transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target layer
# ---------------------------------------------------------------------------


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional feature layer of MobileNetV3-Small.

    For torchvision ``mobilenet_v3_small``, this is ``model.features[-1]``
    — the final 1×1 ``Conv2dNormActivation`` (output 576 channels, 7×7
    spatial for 224×224 input).
    """
    return model.features[-1]


# ---------------------------------------------------------------------------
# Grad-CAM engine
# ---------------------------------------------------------------------------


class GradCAM:
    """Hook-based Grad-CAM for a single image.

    Captures activations and gradients from a target layer, then produces
    a class-discriminative heatmap using the standard Grad-CAM algorithm
    (Selvaraju et al., 2017).

    Parameters
    ----------
    model : nn.Module
        Model (MobileNetV3-Small built by :func:`src.model.build_model`).
    target_layer : nn.Module, optional
        Layer to hook. Defaults to ``model.features[-1]``.

    Important
    ---------
    Always call :meth:`remove_hooks` when done, or use the engine as a
    context manager.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer or get_gradcam_target_layer(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self._bwd_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    # -- hooks ---------------------------------------------------------------

    def _save_activation(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _save_gradient(self, module: nn.Module, grad_input: Any, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    # -- core ----------------------------------------------------------------

    def __call__(
        self,
        img_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        """Run Grad-CAM on a single preprocessed image tensor.

        Parameters
        ----------
        img_tensor : torch.Tensor
            Preprocessed image tensor of shape ``(1, 3, H, W)``.
        class_idx : int, optional
            Class index to explain. If ``None``, uses the model's predicted
            class (``argmax`` of logits).

        Returns
        -------
        (cam, class_idx, logits) : tuple
            - ``cam``: normalised heatmap as ``(H, W)`` float array in [0, 1].
            - ``class_idx``: int, class index that was explained.
            - ``logits``: ``(1, num_classes)`` tensor (reuse for softmax/probs
              to avoid a second forward pass).
        """
        self.model.zero_grad(set_to_none=True)

        # Forward pass (gradients ENABLED — required for backward hook)
        logits = self.model(img_tensor)  # (1, num_classes)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire. Check that the target layer "
                "is in the forward path."
            )

        # Global-average-pool the gradients to get channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations, then ReLU
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # Normalise to [0, 1]
        cam_max = float(cam.max())
        if cam_max > 0:
            cam = cam / cam_max

        return cam, class_idx, logits

    # -- cleanup -------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove forward and backward hooks from the model."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation and convert to ``(H, W, 3)`` uint8 array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def generate_overlay(
    original: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Alpha-blend a heatmap onto the original image.

    Parameters
    ----------
    original : ndarray
        ``(H, W, 3)`` uint8 RGB image.
    cam : ndarray
        ``(H, W)`` normalised heatmap in [0, 1].
    alpha : float
        Blend factor for the heatmap (0 = image only, 1 = heatmap only).
    colormap : str
        Matplotlib colormap name.

    Returns
    -------
    ``(H, W, 3)`` uint8 RGB overlay.
    """
    cmap = plt.get_cmap(colormap)
    heatmap_rgb = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap_rgb + (1 - alpha) * original).astype(np.uint8)
    return overlay


def save_gradcam_outputs(
    image_path: str | Path,
    img_tensor: torch.Tensor,
    cam: np.ndarray,
    overlay: np.ndarray,
    output_dir: str | Path = "reports/gradcam",
    stem: str | None = None,
) -> dict[str, Path]:
    """Save heatmap and overlay images to *output_dir*.

    Parameters
    ----------
    image_path : str or Path
        Original image path (used to derive the filename stem).
    img_tensor : torch.Tensor
        Preprocessed tensor ``(1, 3, H, W)`` (unused directly; kept for API).
    cam : ndarray
        ``(H, W)`` normalised heatmap.
    overlay : ndarray
        ``(H, W, 3)`` RGB overlay.
    output_dir : str or Path
        Output directory.
    stem : str, optional
        Filename stem override.

    Returns
    -------
    dict with ``"heatmap"`` and ``"overlay"`` paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stem is None:
        stem = Path(image_path).stem

    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    # Heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cam, cmap="jet")
    ax.axis("off")
    ax.set_title("Grad-CAM")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Overlay
    Image.fromarray(overlay).save(overlay_path)

    logger.info("Grad-CAM saved: %s, %s", heatmap_path, overlay_path)
    return {"heatmap": heatmap_path, "overlay": overlay_path}


# ---------------------------------------------------------------------------
# High-level inference + Grad-CAM
# ---------------------------------------------------------------------------


def predict_image_with_gradcam(
    model: nn.Module,
    image_path: str | Path,
    device: str | torch.device = "cpu",
    image_size: int = 224,
    class_names: list[str] | None = None,
    save_outputs: bool = False,
    output_dir: str | Path = "reports/gradcam",
    gradcam_alpha: float = 0.4,
) -> dict[str, Any]:
    """Run inference + Grad-CAM in a single forward pass.

    Parameters
    ----------
    model : nn.Module
        Trained model (eval mode).
    image_path : str or Path
        Image to classify.
    device : str or torch.device
        Device for inference.
    image_size : int
        Target image size.
    class_names : list of str, optional
        Class names.
    save_outputs : bool
        If True, save heatmap and overlay to *output_dir*.
    output_dir : str or Path
        Directory for saved visualisations.
    gradcam_alpha : float
        Overlay blend factor.

    Returns
    -------
    dict with the standard inference keys plus ``gradcam`` and
    ``gradcam_target``.
    """
    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    model.eval()

    # Load and preprocess
    transform = get_eval_transforms(image_size)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Single forward + backward pass for both logits and Grad-CAM
    with GradCAM(model) as cam_engine:
        cam, class_idx, logits = cam_engine(img_tensor, class_idx=None)

    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    predicted_index = int(probs.argmax())
    confidence = float(probs[predicted_index])

    prob_dict = {name: float(probs[i]) for i, name in enumerate(class_names)}

    result: dict[str, Any] = {
        "predicted_class": class_names[predicted_index],
        "predicted_index": predicted_index,
        "confidence": confidence,
        "probabilities": prob_dict,
        "image_path": str(image_path),
        "gradcam": cam,
        "gradcam_target": class_names[class_idx],
    }

    if save_outputs:
        original = denormalize(img_tensor.squeeze(0))
        overlay = generate_overlay(original, cam, alpha=gradcam_alpha)
        paths = save_gradcam_outputs(
            image_path, img_tensor, cam, overlay, output_dir=output_dir
        )
        result["gradcam_paths"] = {k: str(v) for k, v in paths.items()}

    return result
