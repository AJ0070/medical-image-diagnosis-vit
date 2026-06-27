"""Optuna hyperparameter search entry point."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import click
import optuna
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model, build_loss
from models.trainer import Trainer
from models.metrics import MetricsCalculator
from preprocessing.dataset import DataModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def _suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def objective(trial: optuna.Trial, base_cfg: dict, search_space: dict, dev: torch.device) -> float:
    params = _suggest_params(trial, search_space)

    # Apply suggested params to config
    cfg = _merge(base_cfg, {
        "optimizer": {
            "name": params.get("optimizer", base_cfg.get("optimizer", {}).get("name", "adamw")),
            "lr": params.get("lr", 1e-4),
            "weight_decay": params.get("weight_decay", 1e-4),
        },
        "training": {
            "batch_size": params.get("batch_size", 32),
            "epochs": 10,   # short runs for HPO
            "early_stopping_patience": 5,
            "mixed_precision": True,
        },
        "scheduler": {
            "name": params.get("scheduler", "cosine"),
        },
        "model": {
            "name": params.get("model", base_cfg.get("model", {}).get("name", "vit_b16")),
            "dropout": params.get("dropout", 0.1),
            "image_size": params.get("image_size", 224),
        },
        "loss": {
            "label_smoothing": params.get("label_smoothing", 0.0),
        },
        "logging": {"tensorboard": {"enabled": False}, "wandb": {"enabled": False}},
    })

    data_module = DataModule(cfg)
    data_module.setup()
    num_classes = cfg.get("dataset", {}).get("num_classes", 2)

    model = build_model(cfg, num_classes=num_classes)
    criterion = build_loss(cfg)

    trainer = Trainer(model=model, cfg=cfg, num_classes=num_classes, device=dev)
    train_loader = data_module.train_dataloader(use_sampler=False)
    val_loader = data_module.val_dataloader()

    val_metrics = MetricsCalculator(num_classes)

    for epoch in range(cfg["training"]["epochs"]):
        trainer.train_epoch(train_loader, criterion, val_metrics)
        result = trainer.eval_epoch(val_loader, criterion, val_metrics)
        f1 = result.get("f1", 0.0)
        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return result.get("f1", 0.0)


@click.command()
@click.option("--config", "-c", required=True)
@click.option("--search-config", "-s", default="configs/hparam_search.yaml")
@click.option("--n-trials", "-n", default=None, type=int)
@click.option("--study-name", default=None)
@click.option("--storage", default=None, help="Optuna DB URI (e.g. sqlite:///study.db)")
@click.option("--device", default=None)
def main(config, search_config, n_trials, study_name, storage, device):
    """Run Optuna hyperparameter search."""
    with open(config) as f:
        base_cfg = yaml.safe_load(f)
    with open(search_config) as f:
        search_cfg = yaml.safe_load(f)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_trials = n_trials or search_cfg.get("study", {}).get("n_trials", 20)
    study_name = study_name or search_cfg.get("study", {}).get("name", "medvit")
    search_space = search_cfg.get("search_space", {})

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(t, base_cfg, search_space, dev),
        n_trials=n_trials,
        timeout=search_cfg.get("study", {}).get("timeout"),
        show_progress_bar=True,
    )

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value (F1): {study.best_value:.4f}")
    logger.info(f"Best params:\n{study.best_params}")

    # Save results
    import pandas as pd
    df = study.trials_dataframe()
    df.to_csv("results/hparam_search_results.csv", index=False)

    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("results/optuna_history.png")
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image("results/optuna_importance.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
