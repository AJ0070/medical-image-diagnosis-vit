"""Training entry point — loads config, builds model/data/optimizer, runs trainer."""

import logging
import random
import sys
import os
from pathlib import Path

import click
import numpy as np
import torch
import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model, build_loss
from models.trainer import Trainer
from preprocessing.dataset import DataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load and merge dataset config over base config."""
    base_path = Path(config_path).parent / "base.yaml"
    cfg = {}
    if base_path.exists():
        with open(base_path) as f:
            cfg = yaml.safe_load(f) or {}
    with open(config_path) as f:
        dataset_cfg = yaml.safe_load(f) or {}
    # Deep merge: dataset config overrides base
    _deep_merge(cfg, dataset_cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@click.command()
@click.option("--config", "-c", required=True, help="Path to dataset YAML config")
@click.option("--data-dir", "-d", default=None, help="Override dataset path")
@click.option("--epochs", "-e", default=None, type=int, help="Override number of epochs")
@click.option("--batch-size", "-b", default=None, type=int, help="Override batch size")
@click.option("--lr", default=None, type=float, help="Override learning rate")
@click.option("--model", "-m", default=None, help="Override model name")
@click.option("--resume", "-r", default=None, help="Checkpoint path to resume from")
@click.option("--no-wandb", is_flag=True, help="Disable W&B logging")
@click.option("--device", default=None, help="Device: cuda / cpu")
@click.option(
    "--fast-dev-run",
    is_flag=True,
    help="Smoke-test: 1 epoch, 8 batches, batch-size 4. Verifies the full pipeline quickly.",
)
def main(config, data_dir, epochs, batch_size, lr, model, resume, no_wandb, device, fast_dev_run):
    """Train a medical image classification model."""
    cfg = load_config(config)

    # CLI overrides
    if data_dir:
        cfg.setdefault("dataset", {})["path"] = data_dir
    if epochs:
        cfg.setdefault("training", {})["epochs"] = epochs
    if batch_size:
        cfg.setdefault("training", {})["batch_size"] = batch_size
    if lr:
        cfg.setdefault("optimizer", {})["lr"] = lr
    if model:
        cfg.setdefault("model", {})["name"] = model
    if resume:
        cfg.setdefault("training", {})["resume_from"] = resume
    if no_wandb:
        cfg.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False

    if fast_dev_run:
        logger.info("Fast-dev-run enabled: 1 epoch, 8 batches, batch_size=4")
        cfg.setdefault("training", {}).update({
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "mixed_precision": False,
            "early_stopping_patience": 999,
        })
        cfg.setdefault("logging", {}).update({
            "tensorboard": {"enabled": False},
            "wandb": {"enabled": False},
        })
        cfg["_fast_dev_run_batches"] = 8

    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Device
    if device:
        dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {dev}")

    # Data
    data_module = DataModule(cfg)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Model
    num_classes = cfg.get("dataset", {}).get("num_classes", 2)
    net = build_model(cfg, num_classes=num_classes)

    # Loss
    class_weights = data_module.train_dataset.get_class_weights().to(dev)
    criterion = build_loss(cfg, class_weights=class_weights)

    # Train
    trainer = Trainer(
        model=net,
        cfg=cfg,
        num_classes=num_classes,
        class_names=data_module.class_names,
        device=dev,
        fast_dev_run_batches=cfg.get("_fast_dev_run_batches", 0),
    )
    trainer.fit(train_loader, val_loader, criterion)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
