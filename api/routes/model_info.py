"""Model info and metrics endpoints."""

import os
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.schemas.prediction import ModelInfo, MetricsResponse
from api.dependencies import get_predictor, get_cfg

router = APIRouter()


@router.get("/model-info", response_model=ModelInfo, summary="Model architecture details")
async def model_info(predictor=Depends(get_predictor), cfg=Depends(get_cfg)):
    model = predictor.model
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ModelInfo(
        model_name=cfg.get("model", {}).get("name", "unknown"),
        dataset=cfg.get("project", {}).get("name", "unknown"),
        num_classes=cfg.get("dataset", {}).get("num_classes", 0),
        class_names=cfg.get("dataset", {}).get("classes", []),
        total_params=total,
        trainable_params=trainable,
        image_size=cfg.get("model", {}).get("image_size", 224),
        checkpoint_path=os.environ.get("CHECKPOINT_PATH"),
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Last evaluation metrics")
async def metrics(cfg=Depends(get_cfg)):
    metrics_path = os.path.join(
        cfg.get("project", {}).get("output_dir", "results"), "metrics.csv"
    )
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="No evaluation metrics found. Run evaluate.py first.")
    df = pd.read_csv(metrics_path)
    row = df.iloc[0].to_dict()
    return MetricsResponse(
        model_name=cfg.get("model", {}).get("name", "unknown"),
        dataset=cfg.get("project", {}).get("name", "unknown"),
        accuracy=row.get("accuracy"),
        f1=row.get("f1"),
        roc_auc=row.get("roc_auc"),
        precision=row.get("precision"),
        recall=row.get("recall"),
        additional={k: v for k, v in row.items() if k not in ("accuracy", "f1", "roc_auc", "precision", "recall")},
    )
