"""Health check endpoint."""

import torch
from fastapi import APIRouter, Depends

from api.schemas.prediction import HealthResponse
from api.dependencies import get_predictor, get_cfg

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="API health check")
async def health(predictor=Depends(get_predictor), cfg=Depends(get_cfg)):
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        device=str(predictor.device),
        model_name=cfg.get("model", {}).get("name", "unknown"),
        dataset=cfg.get("project", {}).get("name", "unknown"),
        num_classes=cfg.get("dataset", {}).get("num_classes", 0),
    )
