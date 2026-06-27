"""Compare ViT and EfficientNet architectures on the same dataset."""

import logging
import sys
import time
from pathlib import Path

import click
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_loss
from models.evaluator import Evaluator
from models.metrics import MetricsCalculator
from models.trainer import Trainer
from models.vit import build_vit
from models.efficientnet import build_efficientnet
from preprocessing.dataset import DataModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COMPARE_MODELS = [
    ("vit_b16", "vit"),
    ("efficientnet_b0", "efficientnet"),
    ("efficientnet_b3", "efficientnet"),
]


@click.command()
@click.option("--config", "-c", required=True)
@click.option("--epochs", "-e", default=10, type=int)
@click.option("--output", "-o", default="results/model_comparison.csv")
@click.option("--device", default=None)
def main(config, epochs, output, device):
    """Train and compare multiple architectures, output a comparison CSV."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    cfg["training"]["epochs"] = epochs
    cfg["logging"] = {"tensorboard": {"enabled": False}, "wandb": {"enabled": False}}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = cfg.get("dataset", {}).get("num_classes", 2)
    class_names = cfg.get("dataset", {}).get("classes", [])

    data_module = DataModule(cfg)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    criterion = build_loss(cfg)

    rows = []
    for model_name, family in COMPARE_MODELS:
        logger.info(f"\n{'='*50}\nEvaluating {model_name}\n{'='*50}")
        cfg["model"]["name"] = model_name
        if family == "vit":
            model = build_vit(cfg, num_classes)
        else:
            model = build_efficientnet(cfg, num_classes)

        trainer = Trainer(model=model, cfg=cfg, num_classes=num_classes, device=dev)
        val_metrics = MetricsCalculator(num_classes, class_names)
        t0 = time.time()
        trainer.fit(train_loader, val_loader, criterion)
        train_time = time.time() - t0

        # Inference timing
        model.eval().to(dev)
        dummy = torch.randn(1, 3, cfg["model"].get("image_size", 224),
                           cfg["model"].get("image_size", 224)).to(dev)
        with torch.no_grad():
            t1 = time.time()
            for _ in range(100):
                model(dummy)
            infer_ms = (time.time() - t1) / 100 * 1000

        evaluator = Evaluator(model, num_classes, class_names, dev, output_dir=f"results/{model_name}")
        metrics = evaluator.evaluate(val_loader)

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        rows.append({
            "model": model_name,
            "params_M": round(total_params, 2),
            "train_time_s": round(train_time, 1),
            "infer_ms": round(infer_ms, 2),
            **{k: round(v, 4) for k, v in metrics.items()},
        })

    df = pd.DataFrame(rows).set_index("model")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output)
    print("\n=== Model Comparison ===")
    print(df.to_string())


if __name__ == "__main__":
    main()
