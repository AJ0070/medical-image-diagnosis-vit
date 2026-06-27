"""Export trained models to ONNX and TorchScript formats."""

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


@click.command()
@click.option("--config", "-c", required=True)
@click.option("--checkpoint", "-k", required=True)
@click.option("--output-dir", "-o", default="checkpoints/exported")
@click.option("--format", "-f", "fmt", type=click.Choice(["onnx", "torchscript", "both"]), default="both")
@click.option("--device", default=None)
def main(config, checkpoint, output_dir, fmt, device):
    """Export model to ONNX and/or TorchScript."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = cfg.get("dataset", {}).get("num_classes", 2)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, num_classes=num_classes)
    predictor = Predictor(model=model, cfg=cfg, checkpoint_path=checkpoint, device=dev)

    project_name = cfg.get("project", {}).get("name", "model")

    if fmt in ("onnx", "both"):
        predictor.export_onnx(f"{output_dir}/{project_name}.onnx")
    if fmt in ("torchscript", "both"):
        predictor.export_torchscript(f"{output_dir}/{project_name}.pt")


if __name__ == "__main__":
    main()
