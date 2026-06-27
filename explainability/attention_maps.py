"""Attention rollout visualization for Vision Transformer models."""

import math
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class ViTAttentionVisualizer:
    """Extract and visualize attention weights from ViT models."""

    def __init__(self, model: nn.Module, discard_ratio: float = 0.9) -> None:
        self.model = model
        self.discard_ratio = discard_ratio
        self._attentions: List[torch.Tensor] = []
        self._hooks = []

    def _register_hooks(self) -> None:
        raw = self.model.module if hasattr(self.model, "module") else self.model
        for block in raw.backbone.blocks:
            h = block.attn.register_forward_hook(self._save_attention)
            self._hooks.append(h)

    def _save_attention(self, module, input, output) -> None:
        # timm ViT attention returns (output, attn_weights) or just output
        # We need to hook into the softmax inside the attention module
        pass

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def get_attention_rollout(
        self, input_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute attention rollout for a ViT input.

        Returns:
            rollout: (H, W) attention map
            raw_attn: attention from last layer (num_heads, N, N)
            mean_attn: mean over heads from last layer (N, N)
        """
        raw = self.model.module if hasattr(self.model, "module") else self.model
        backbone = raw.backbone

        # Forward pass collecting attention weights from all layers
        attention_maps = []

        def hook_fn(module, input, output):
            # For timm ViT, attn.softmax is inside; we capture by hooking the whole Attention
            # Here we reconstruct by accessing qkv directly
            pass

        # Use timm's built-in attention weight extraction
        B = input_tensor.shape[0]
        x = backbone.patch_embed(input_tensor)
        cls_tokens = backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + backbone.pos_embed
        x = backbone.pos_drop(x)

        for block in backbone.blocks:
            # Manually extract attention from each block
            norm_x = block.norm1(x)
            B_b, N, C = norm_x.shape
            qkv = block.attn.qkv(norm_x).reshape(B_b, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = block.attn.scale
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach().cpu())
            # Continue normal block forward
            x = block(x)

        x = backbone.norm(x)

        # Attention rollout
        rollout = self._rollout(attention_maps)

        # (num_heads, N, N) for last layer
        raw_attn = attention_maps[-1][0].numpy()
        mean_attn = raw_attn.mean(axis=0)

        num_patches = x.shape[1] - 1
        grid = int(math.sqrt(num_patches))
        rollout_map = rollout[1:].reshape(grid, grid)

        return rollout_map, raw_attn, mean_attn

    def _rollout(self, attentions: List[torch.Tensor]) -> np.ndarray:
        """Compute attention rollout (Abnar & Zuidema 2020)."""
        result = torch.eye(attentions[0].shape[-1])
        for attn in attentions:
            attn_heads_mean = attn[0].mean(dim=0)  # (N, N)
            # Add residual
            attn_rollout = attn_heads_mean + torch.eye(attn_heads_mean.shape[-1])
            attn_rollout = attn_rollout / attn_rollout.sum(dim=-1, keepdim=True)
            # Discard low-attention connections
            flat = attn_rollout.flatten()
            threshold = flat.kthvalue(int(self.discard_ratio * flat.shape[0])).values
            attn_rollout[attn_rollout < threshold] = 0
            attn_rollout = attn_rollout / (attn_rollout.sum(dim=-1, keepdim=True) + 1e-8)
            result = result @ attn_rollout
        # cls token row
        return result[0].numpy()

    def visualize(
        self,
        image_rgb: np.ndarray,
        rollout: np.ndarray,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Overlay attention rollout on original image."""
        h, w = image_rgb.shape[:2]
        mask = cv2.resize(rollout, (w, h))
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HOT)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.clip(0.6 * image_rgb + 0.4 * heatmap, 0, 255).astype(np.uint8)

        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image_rgb)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(mask, cmap="hot")
            axes[1].set_title("Attention Rollout")
            axes[1].axis("off")
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        return overlay

    def plot_attention_heads(
        self,
        raw_attn: np.ndarray,
        num_heads_to_show: int = 4,
        patch_grid_size: int = 14,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot individual attention heads from the last ViT layer."""
        num_heads = min(num_heads_to_show, raw_attn.shape[0])
        fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
        if num_heads == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            cls_attn = raw_attn[i, 0, 1:]  # cls token attending to patches
            attn_map = cls_attn.reshape(patch_grid_size, patch_grid_size)
            ax.imshow(attn_map, cmap="viridis")
            ax.set_title(f"Head {i+1}")
            ax.axis("off")
        plt.suptitle("ViT Last-Layer Attention Heads (CLS → Patches)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
