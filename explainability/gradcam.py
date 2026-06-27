"""Grad-CAM and Grad-CAM++ for CNN-based models (EfficientNet etc.)."""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Returns:
            cam: normalized heatmap (H, W) in [0, 1]
            predicted_class: int
            probs: softmax probabilities
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        predicted_class = probs.argmax()

        target = target_class if target_class is not None else predicted_class

        self.model.zero_grad()
        score = output[0, target]
        score.backward()

        gradients = self._gradients[0]         # (C, H, W)
        activations = self._activations[0]     # (C, H, W)

        # Global average pool gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
        cam = (weights * activations).sum(dim=0)              # (H, W)
        cam = F.relu(cam).cpu().numpy()

        cam = self._normalize(cam)
        return cam, int(predicted_class), probs

    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def overlay_on_image(
        self,
        image_rgb: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay CAM heatmap on an RGB image."""
        h, w = image_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.clip(
            (1 - alpha) * image_rgb.astype(float) + alpha * heatmap.astype(float), 0, 255
        ).astype(np.uint8)
        return overlay

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()

    def __del__(self):
        self.remove_hooks()


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ with second-order gradient weighting."""

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        predicted_class = probs.argmax()
        target = target_class if target_class is not None else predicted_class

        self.model.zero_grad()
        score = output[0, target]
        score.backward()

        gradients = self._gradients[0]         # (C, H, W)
        activations = self._activations[0]     # (C, H, W)

        # Grad-CAM++ alpha computation
        grad_sq = gradients ** 2
        grad_cu = gradients ** 3
        sum_acts = activations.sum(dim=(1, 2), keepdim=True)
        alpha_denom = 2 * grad_sq + sum_acts * grad_cu
        alpha_denom = torch.where(
            alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom)
        )
        alpha = grad_sq / alpha_denom
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)

        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam).cpu().numpy()
        cam = self._normalize(cam)
        return cam, int(predicted_class), probs


def get_target_layer(model: nn.Module) -> nn.Module:
    """Auto-detect the last conv/feature layer for Grad-CAM."""
    model_name = getattr(model, "model_name", "")
    raw = model.module if hasattr(model, "module") else model

    if "efficientnet" in model_name:
        # Last conv block in EfficientNet backbone
        blocks = list(raw.backbone.blocks)
        for block in reversed(blocks):
            for layer in reversed(list(block.modules())):
                if isinstance(layer, nn.Conv2d):
                    return layer
    # Fallback: find last conv2d
    last_conv = None
    for layer in raw.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv = layer
    if last_conv:
        return last_conv
    raise ValueError("No suitable target layer found for Grad-CAM")
