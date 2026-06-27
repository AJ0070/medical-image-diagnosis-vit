"""Evaluation entry point — loads a checkpoint and runs full evaluation on test set."""

import logging
import sys
from pathlib import Path

import click
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model
from models.evaluator import Evaluator
from preprocessing.dataset import DataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", required=True, help="Path to dataset config YAML")
@click.option("--checkpoint", "-k", required=True, help="Path to model checkpoint .pth")
@click.option("--output-dir", "-o", default="results", help="Directory for evaluation outputs")
@click.option("--split", default="test", type=click.Choice(["val", "test"]), help="Eval split")
@click.option("--device", default=None)
def main(config, checkpoint, output_dir, split, device):
    """Evaluate a trained model and generate metrics/plots."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = cfg.get("dataset", {}).get("num_classes", 2)
    class_names = cfg.get("dataset", {}).get("classes", [])

    model = build_model(cfg, num_classes=num_classes)
    state = torch.load(checkpoint, map_location=dev)
    model.load_state_dict(state["model_state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint} (epoch {state.get('epoch', '?')})")

    data_module = DataModule(cfg)
    data_module.setup()
    loader = data_module.test_dataloader() if split == "test" else data_module.val_dataloader()

    evaluator = Evaluator(
        model=model,
        num_classes=num_classes,
        class_names=class_names,
        device=dev,
        output_dir=output_dir,
    )
    metrics = evaluator.evaluate(loader)

    print("\n=== Final Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()
