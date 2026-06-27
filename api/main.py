"""FastAPI application entry point for medical image diagnosis."""

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api import dependencies
from api.routes import predict, health, model_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/chest_xray.yaml")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/chest-xray-pneumonia/best_model.pth")
DEVICE = os.environ.get("DEVICE", None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    dependencies.initialize(CONFIG_PATH, CHECKPOINT_PATH, DEVICE)
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Medical Image Diagnosis API",
    description=(
        "Production-grade REST API for classifying medical images using "
        "Vision Transformers and EfficientNet with Grad-CAM explainability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(model_info.router, tags=["Model"])


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
