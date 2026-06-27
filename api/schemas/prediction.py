"""Pydantic schemas for API request/response validation."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    model_name: str
    dataset: str
    has_gradcam: bool = False
    has_attention: bool = False


class ModelInfo(BaseModel):
    model_name: str
    dataset: str
    num_classes: int
    class_names: List[str]
    total_params: int
    trainable_params: int
    image_size: int
    checkpoint_path: Optional[str] = None


class MetricsResponse(BaseModel):
    model_name: str
    dataset: str
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    additional: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str
    dataset: str
    num_classes: int
