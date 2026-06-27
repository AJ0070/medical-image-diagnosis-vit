"""Vision Transformer models using timm — ViT-B/16 and ViT-L/16."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

# Supported ViT variants
VIT_VARIANTS = {
    "vit_b16": "vit_base_patch16_224",
    "vit_b32": "vit_base_patch32_224",
    "vit_l16": "vit_large_patch16_224",
    "vit_l32": "vit_large_patch32_224",
}


class ViTClassifier(nn.Module):
    """ViT backbone with a custom classification head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.1,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        timm_name = VIT_VARIANTS.get(model_name, model_name)
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,          # remove default head
            img_size=image_size,
            drop_rate=dropout,
            attn_drop_rate=dropout / 2,
        )
        embed_dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.num_classes = num_classes
        self.model_name = model_name
        logger.info(f"Built {model_name} with {self.count_params():,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return output and last-layer attention weights for visualization."""
        with torch.no_grad():
            # Access last transformer block's attention
            blocks = self.backbone.blocks
            B, C, H, W = x.shape
            x_feat = self.backbone.patch_embed(x)
            cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
            x_feat = torch.cat((cls_tokens, x_feat), dim=1)
            x_feat = x_feat + self.backbone.pos_embed
            x_feat = self.backbone.pos_drop(x_feat)

            attn_weights = None
            for i, block in enumerate(blocks):
                if i == len(blocks) - 1:
                    # Hook last attention block
                    attn_weights = block.attn.softmax(
                        block.attn.proj_drop(
                            block.attn.attn_drop(
                                block.attn.scale
                                * (block.attn.q(block.norm1(x_feat))
                                   @ block.attn.k(block.norm1(x_feat)).transpose(-2, -1))
                            )
                        )
                    )
                x_feat = block(x_feat)

            x_feat = self.backbone.norm(x_feat)
            cls_out = x_feat[:, 0]
            logits = self.head(cls_out)
        return logits, attn_weights

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info(f"Froze backbone — training head only")

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Unfroze backbone — full fine-tuning")

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_vit(cfg: dict, num_classes: int) -> ViTClassifier:
    model_cfg = cfg.get("model", {})
    return ViTClassifier(
        model_name=model_cfg.get("name", "vit_b16"),
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.1),
        image_size=model_cfg.get("image_size", 224),
    )
