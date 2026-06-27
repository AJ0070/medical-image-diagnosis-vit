"""Prediction endpoint — accepts image upload, returns classification + visualizations."""

import base64
import io
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from api.schemas.prediction import PredictionResponse
from api.dependencies import get_predictor

logger = logging.getLogger(__name__)
router = APIRouter()


def _encode_image(image: Optional[np.ndarray]) -> Optional[str]:
    """Encode numpy RGB image to base64 PNG string."""
    if image is None:
        return None
    _, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")


@router.post("/predict", response_model=PredictionResponse, summary="Classify a medical image")
async def predict(
    file: UploadFile = File(..., description="Medical image (JPEG/PNG)"),
    gradcam: bool = Query(True, description="Generate Grad-CAM heatmap"),
    attention: bool = Query(True, description="Generate attention map (ViT only)"),
    tta: bool = Query(False, description="Use test-time augmentation"),
    predictor=Depends(get_predictor),
):
    """Upload a medical image and receive a diagnosis prediction."""
    if file.content_type not in ("image/jpeg", "image/png", "image/bmp"):
        raise HTTPException(status_code=415, detail="Unsupported image format")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        result = predictor.predict(
            image,
            generate_gradcam=gradcam,
            generate_attention=attention,
            use_tta=tta,
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    cfg = predictor.cfg
    response_data = result.to_dict()
    response_data["model_name"] = cfg.get("model", {}).get("name", "unknown")
    response_data["dataset"] = cfg.get("project", {}).get("name", "unknown")
    response_data["has_gradcam"] = result.gradcam_overlay is not None
    response_data["has_attention"] = result.attention_overlay is not None

    # Attach encoded visualizations as extra fields
    extras = {}
    if result.gradcam_overlay is not None:
        extras["gradcam_b64"] = _encode_image(result.gradcam_overlay)
    if result.attention_overlay is not None:
        extras["attention_b64"] = _encode_image(result.attention_overlay)

    return JSONResponse(content={**response_data, **extras})
