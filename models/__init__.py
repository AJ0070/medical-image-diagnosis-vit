from models.vit import build_vit
from models.efficientnet import build_efficientnet
from models.losses import build_loss
from models.metrics import MetricsCalculator

__all__ = ["build_vit", "build_efficientnet", "build_loss", "MetricsCalculator"]


def build_model(cfg: dict, num_classes: int):
    """Factory: pick architecture from config."""
    name = cfg.get("model", {}).get("name", "vit_b16")
    if name.startswith("vit"):
        return build_vit(cfg, num_classes)
    elif name.startswith("efficientnet"):
        return build_efficientnet(cfg, num_classes)
    raise ValueError(f"Unknown model: {name}")
