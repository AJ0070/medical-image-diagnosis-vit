"""Single-image inference with Grad-CAM visualization."""

import logging
import sys
from pathlib import Path

import click
import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model
from preprocessing.augmentation import build_augmentation_pipeline
from preprocessing.transforms import load_image, denormalize
from explainability.gradcam import GradCAMPlusPlus, get_target_layer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", required=True)
@click.option("--checkpoint", "-k", required=True)
@click.option("--input", "-i", "image_path", required=True, help="Path to input image")
@click.option("--output-dir", "-o", default="results/inference")
@click.option("--device", default=None)
def main(config, checkpoint, image_path, output_dir, device):
    """Run inference on a single image and generate Grad-CAM visualization."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = cfg.get("dataset", {}).get("num_classes", 2)
    class_names = cfg.get("dataset", {}).get("classes", [])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model(cfg, num_classes=num_classes)
    state = torch.load(checkpoint, map_location=dev)
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(dev)
    logger.info(f"Loaded checkpoint: {checkpoint}")

    # Preprocess
    image = load_image(image_path)
    transform = build_augmentation_pipeline(cfg, split="val")
    tensor = transform(image=image)["image"].unsqueeze(0).to(dev)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(probs.argmax())
    pred_label = class_names[pred_class] if class_names else str(pred_class)
    confidence = float(probs[pred_class])

    print(f"\nPrediction : {pred_label}")
    print(f"Confidence : {confidence:.1%}")
    print("Probabilities:")
    for i, p in enumerate(probs):
        label = class_names[i] if class_names else str(i)
        bar = "█" * int(p * 30)
        print(f"  {label:20s} {p:.3f} {bar}")

    # Grad-CAM (EfficientNet only)
    model_name = cfg.get("model", {}).get("name", "")
    if "efficientnet" in model_name:
        try:
            target_layer = get_target_layer(model)
            gradcam = GradCAMPlusPlus(model, target_layer)
            cam, _, _ = gradcam.generate(tensor)
            overlay = gradcam.overlay_on_image(image, cam)
            gradcam.remove_hooks()

            stem = Path(image_path).stem
            cv2.imwrite(str(out_dir / f"{stem}_original.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"{stem}_gradcam.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            logger.info(f"Grad-CAM saved to {out_dir}")
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")
    else:
        logger.info("Grad-CAM not available for ViT — use EfficientNet for visualization")


if __name__ == "__main__":
    main()
