"""FastAPI dependency injection — model and config singleton."""

import logging
import os
from functools import lru_cache
from typing import Optional

import torch
import yaml

from models import build_model
from inference.predictor import Predictor

logger = logging.getLogger(__name__)

_predictor: Optional[Predictor] = None
_cfg: Optional[dict] = None


def initialize(config_path: str, checkpoint_path: str, device: Optional[str] = None) -> None:
    """Called at startup to load model into memory."""
    global _predictor, _cfg
    with open(config_path) as f:
        _cfg = yaml.safe_load(f)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = _cfg.get("dataset", {}).get("num_classes", 2)
    model = build_model(_cfg, num_classes=num_classes)

    _predictor = Predictor(
        model=model,
        cfg=_cfg,
        checkpoint_path=checkpoint_path,
        device=dev,
    )
    logger.info(f"Model loaded on {dev}")


def initialize_config_only(config_path: str) -> None:
    """Load config without a model — for demo/no-checkpoint mode."""
    global _cfg
    with open(config_path) as f:
        _cfg = yaml.safe_load(f)


def get_predictor() -> Predictor:
    from fastapi import HTTPException
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model and provide a checkpoint path to enable predictions.",
        )
    return _predictor


def get_cfg() -> dict:
    from fastapi import HTTPException
    if _cfg is None:
        raise HTTPException(status_code=503, detail="Config not loaded.")
    return _cfg
