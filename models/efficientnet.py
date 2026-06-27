"""EfficientNet models — B0 and B3 — via timm with custom classification head."""

import logging
from typing import Dict

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

EFFICIENTNET_VARIANTS: Dict[str, str] = {
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
}


class EfficientNetClassifier(nn.Module):
    """EfficientNet backbone with a custom classification head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        timm_name = EFFICIENTNET_VARIANTS.get(model_name, model_name)
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if self._needs_pool() else nn.Identity(),
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(in_features // 2, num_classes),
        )
        self.num_classes = num_classes
        self.model_name = model_name
        logger.info(f"Built {model_name} with {self.count_params():,} parameters")

    def _needs_pool(self) -> bool:
        return False  # timm backbone already pools

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature maps before global pooling for Grad-CAM."""
        return self.backbone.forward_features(x)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Froze EfficientNet backbone")

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 0) -> None:
        """Unfreeze the full backbone or only the last N blocks."""
        if unfreeze_last_n_blocks == 0:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Unfroze all EfficientNet layers")
            return

        blocks = list(self.backbone.blocks)
        for block in blocks[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True
        logger.info(f"Unfroze last {unfreeze_last_n_blocks} EfficientNet blocks")

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_efficientnet(cfg: dict, num_classes: int) -> EfficientNetClassifier:
    model_cfg = cfg.get("model", {})
    return EfficientNetClassifier(
        model_name=model_cfg.get("name", "efficientnet_b0"),
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2),
    )
