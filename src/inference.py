"""
AgriMind AI — Inference Module.

Simple inference function: image path -> preprocessing -> model ->
softmax -> prediction -> confidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from src.dataset import IDX_TO_CLASS, NUM_CLASSES, get_eval_transforms
from src.model import load_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_image(
    model: nn.Module,
    image_path: str | Path,
    device: str | torch.device = "cpu",
    image_size: int = 224,
    class_names: list[str] | None = None,
    gradcam: bool = False,
    save_gradcam: bool = False,
    gradcam_output_dir: str | Path = "reports/gradcam",
    gradcam_alpha: float = 0.4,
) -> dict[str, Any]:
    """Run inference on a single image.

    Parameters
    ----------
    model : nn.Module
        Trained model (already loaded and in eval mode).
    image_path : str or Path
        Path to the image file.
    device : str or torch.device
        Device for inference.
    image_size : int
        Target image size for preprocessing.
    class_names : list of str, optional
        Class names. Defaults to IDX_TO_CLASS mapping.
    gradcam : bool
        If True, also compute a Grad-CAM heatmap and overlay. Adds
        ``gradcam`` and ``gradcam_target`` keys to the result dict.
    save_gradcam : bool
        If True (and *gradcam* is True), save the heatmap and overlay
        to *gradcam_output_dir*.
    gradcam_output_dir : str or Path
        Directory for saved Grad-CAM visualisations.
    gradcam_alpha : float
        Overlay blend factor (0 = image only, 1 = heatmap only).

    Returns
    -------
    dict with:
        - ``predicted_class``: str (e.g., "Healthy")
        - ``predicted_index``: int
        - ``confidence``: float in [0, 1]
        - ``probabilities``: dict mapping class name -> probability
        - ``image_path``: str
        - ``gradcam`` (if *gradcam* is True): normalised heatmap array
        - ``gradcam_target`` (if *gradcam* is True): class name explained
    """
    if gradcam:
        from src.gradcam import predict_image_with_gradcam

        return predict_image_with_gradcam(
            model,
            image_path,
            device=device,
            image_size=image_size,
            class_names=class_names,
            save_outputs=save_gradcam,
            output_dir=gradcam_output_dir,
            gradcam_alpha=gradcam_alpha,
        )
    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    model.eval()

    # Load and preprocess
    transform = get_eval_transforms(image_size)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    predicted_index = int(probs.argmax())
    confidence = float(probs[predicted_index])

    prob_dict = {
        name: float(probs[i]) for i, name in enumerate(class_names)
    }

    return {
        "predicted_class": class_names[predicted_index],
        "predicted_index": predicted_index,
        "confidence": confidence,
        "probabilities": prob_dict,
        "image_path": str(image_path),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python -m src.inference``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="AgriMind AI — Image Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image", help="Path to image file",
    )
    parser.add_argument(
        "--checkpoint", default="models/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image size",
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Compute Grad-CAM heatmap and overlay",
    )
    parser.add_argument(
        "--save-gradcam", action="store_true",
        help="Save Grad-CAM heatmap and overlay images",
    )
    parser.add_argument(
        "--gradcam-output-dir", default="reports/gradcam",
        help="Directory for saved Grad-CAM outputs",
    )
    parser.add_argument(
        "--gradcam-alpha", type=float, default=0.4,
        help="Grad-CAM overlay blend factor",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    model = load_checkpoint(args.checkpoint, device=device)

    # Predict
    result = predict_image(
        model, args.image, device=device, image_size=args.image_size,
        gradcam=args.gradcam,
        save_gradcam=args.save_gradcam,
        gradcam_output_dir=args.gradcam_output_dir,
        gradcam_alpha=args.gradcam_alpha,
    )

    # Print result
    print(f"Image: {result['image_path']}")
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls}: {prob:.4f}")
    if args.gradcam:
        print(f"Grad-CAM target: {result['gradcam_target']}")
        if "gradcam_paths" in result:
            print(f"Grad-CAM saved: {result['gradcam_paths']}")


if __name__ == "__main__":
    main()
