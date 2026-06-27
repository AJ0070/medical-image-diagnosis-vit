"""Inference CLI — single image or batch folder prediction with optional Grad-CAM output."""

import csv
import logging
import sys
from pathlib import Path

import click
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model
from inference.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@click.command()
@click.option("--config", "-c", required=True, help="Dataset config YAML path")
@click.option("--checkpoint", "-k", required=True, help="Model checkpoint .pth")
@click.option("--input", "-i", required=True, help="Image path or directory")
@click.option("--output-dir", "-o", default="results/inference", help="Output directory")
@click.option("--no-gradcam", is_flag=True, help="Skip Grad-CAM generation")
@click.option("--no-attention", is_flag=True, help="Skip attention map generation")
@click.option("--tta", is_flag=True, help="Enable test-time augmentation")
@click.option("--batch-size", "-b", default=32, type=int)
@click.option("--device", default=None)
def main(config, checkpoint, input, output_dir, no_gradcam, no_attention, tta, batch_size, device):
    """Run medical image inference with Grad-CAM and attention visualization."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = cfg.get("dataset", {}).get("num_classes", 2)
    class_names = cfg.get("dataset", {}).get("classes", [])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, num_classes=num_classes)
    predictor = Predictor(model=model, cfg=cfg, checkpoint_path=checkpoint, device=dev)

    input_path = Path(input)

    if input_path.is_file():
        # Single image
        result = predictor.predict(
            str(input_path),
            generate_gradcam=not no_gradcam,
            generate_attention=not no_attention,
            use_tta=tta,
        )
        _print_result(input_path.name, result)
        _save_visuals(result, out_dir / input_path.stem)

    elif input_path.is_dir():
        # Batch directory
        image_paths = [str(p) for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
        logger.info(f"Found {len(image_paths)} images in {input_path}")
        results = predictor.predict_batch(image_paths, batch_size=batch_size)
        csv_path = out_dir / "predictions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "predicted_label", "confidence"] + class_names
            )
            writer.writeheader()
            for path, res in zip(image_paths, results):
                row = {
                    "filename": Path(path).name,
                    "predicted_label": res.predicted_label,
                    "confidence": res.confidence,
                }
                row.update(res.probabilities)
                writer.writerow(row)
        logger.info(f"Predictions saved to {csv_path}")
    else:
        logger.error(f"Input not found: {input}")


def _print_result(name, result):
    print(f"\n{'='*50}")
    print(f"Image: {name}")
    print(f"Prediction: {result.predicted_label} ({result.confidence:.1%})")
    print("Probabilities:")
    for cls, prob in result.probabilities.items():
        bar = "█" * int(prob * 30)
        print(f"  {cls:20s} {prob:.3f} {bar}")


def _save_visuals(result, out_prefix):
    import cv2
    if result.original_image is not None:
        cv2.imwrite(str(out_prefix) + "_original.png",
                    cv2.cvtColor(result.original_image, cv2.COLOR_RGB2BGR))
    if result.gradcam_overlay is not None:
        cv2.imwrite(str(out_prefix) + "_gradcam.png",
                    cv2.cvtColor(result.gradcam_overlay, cv2.COLOR_RGB2BGR))
    if result.attention_overlay is not None:
        cv2.imwrite(str(out_prefix) + "_attention.png",
                    cv2.cvtColor(result.attention_overlay, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
