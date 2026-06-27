"""End-to-end inference pipeline: preprocess → predict → Grad-CAM → attention map."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image

from explainability.gradcam import GradCAM, GradCAMPlusPlus, get_target_layer
from explainability.attention_maps import ViTAttentionVisualizer
from preprocessing.augmentation import build_augmentation_pipeline, build_tta_pipeline
from preprocessing.transforms import load_image, denormalize

logger = logging.getLogger(__name__)


class PredictionResult:
    """Structured result from a single inference."""

    def __init__(
        self,
        predicted_class: int,
        predicted_label: str,
        confidence: float,
        probabilities: Dict[str, float],
        gradcam_heatmap: Optional[np.ndarray] = None,
        gradcam_overlay: Optional[np.ndarray] = None,
        attention_map: Optional[np.ndarray] = None,
        attention_overlay: Optional[np.ndarray] = None,
        original_image: Optional[np.ndarray] = None,
    ) -> None:
        self.predicted_class = predicted_class
        self.predicted_label = predicted_label
        self.confidence = confidence
        self.probabilities = probabilities
        self.gradcam_heatmap = gradcam_heatmap
        self.gradcam_overlay = gradcam_overlay
        self.attention_map = attention_map
        self.attention_overlay = attention_overlay
        self.original_image = original_image

    def to_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "confidence": round(float(self.confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
        }


class Predictor:
    """Full inference engine with optional Grad-CAM and attention visualization."""

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names: List[str] = cfg.get("dataset", {}).get("classes", [])
        self.num_classes: int = cfg.get("dataset", {}).get("num_classes", 2)
        self.image_size: int = cfg.get("model", {}).get("image_size", 224)
        self.model_name: str = cfg.get("model", {}).get("name", "")

        self.model = model.to(self.device)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        self.model.eval()

        self.transform = build_augmentation_pipeline(cfg, split="val")
        self.tta_transforms = build_tta_pipeline(
            self.image_size,
            cfg.get("data", {}).get("mean", [0.485, 0.456, 0.406]),
            cfg.get("data", {}).get("std", [0.229, 0.224, 0.225]),
        )

        # Grad-CAM setup (CNN only)
        self._gradcam: Optional[GradCAM] = None
        if "efficientnet" in self.model_name:
            try:
                target_layer = get_target_layer(self.model)
                self._gradcam = GradCAMPlusPlus(self.model, target_layer)
            except Exception as e:
                logger.warning(f"Could not set up Grad-CAM: {e}")

        # ViT attention visualizer
        self._attn_viz: Optional[ViTAttentionVisualizer] = None
        if "vit" in self.model_name:
            self._attn_viz = ViTAttentionVisualizer(self.model)

    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        model_state = state.get("model_state_dict", state)
        self.model.load_state_dict(model_state)
        logger.info(f"Loaded checkpoint: {path}")

    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        generate_gradcam: bool = True,
        generate_attention: bool = True,
        use_tta: bool = False,
    ) -> PredictionResult:
        """Run inference on a single image."""
        # Load image
        if isinstance(image, str):
            original = load_image(image)
        elif isinstance(image, Image.Image):
            original = np.array(image.convert("RGB"))
        else:
            original = image.copy()

        tensor = self._preprocess(original)

        if use_tta and self.cfg.get("inference", {}).get("tta_enabled", False):
            probs = self._tta_predict(original)
        else:
            with torch.no_grad():
                logits = self.model(tensor.unsqueeze(0).to(self.device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class = int(probs.argmax())
        pred_label = self.class_names[pred_class] if self.class_names else str(pred_class)
        prob_dict = {
            (self.class_names[i] if self.class_names else str(i)): float(p)
            for i, p in enumerate(probs)
        }

        # Grad-CAM
        gradcam_heatmap, gradcam_overlay = None, None
        if generate_gradcam and self._gradcam is not None:
            cam, _, _ = self._gradcam.generate(tensor.unsqueeze(0).to(self.device))
            gradcam_heatmap = cam
            gradcam_overlay = self._gradcam.overlay_on_image(original, cam)

        # ViT Attention
        attn_map, attn_overlay = None, None
        if generate_attention and self._attn_viz is not None:
            rollout, _, _ = self._attn_viz.get_attention_rollout(
                tensor.unsqueeze(0).to(self.device)
            )
            attn_map = rollout
            attn_overlay = self._attn_viz.visualize(original, rollout)

        return PredictionResult(
            predicted_class=pred_class,
            predicted_label=pred_label,
            confidence=float(probs[pred_class]),
            probabilities=prob_dict,
            gradcam_heatmap=gradcam_heatmap,
            gradcam_overlay=gradcam_overlay,
            attention_map=attn_map,
            attention_overlay=attn_overlay,
            original_image=original,
        )

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32,
    ) -> List[PredictionResult]:
        """Run inference over a list of image paths."""
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            tensors = []
            originals = []
            for p in batch_paths:
                orig = load_image(p)
                tensors.append(self._preprocess(orig))
                originals.append(orig)
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
                probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            for orig, probs in zip(originals, probs_batch):
                pred_class = int(probs.argmax())
                pred_label = self.class_names[pred_class] if self.class_names else str(pred_class)
                prob_dict = {
                    (self.class_names[j] if self.class_names else str(j)): float(p)
                    for j, p in enumerate(probs)
                }
                results.append(
                    PredictionResult(
                        predicted_class=pred_class,
                        predicted_label=pred_label,
                        confidence=float(probs[pred_class]),
                        probabilities=prob_dict,
                        original_image=orig,
                    )
                )
        return results

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        aug = self.transform(image=image)
        return aug["image"]

    def _tta_predict(self, image: np.ndarray) -> np.ndarray:
        all_probs = []
        with torch.no_grad():
            for t in self.tta_transforms:
                aug = t(image=image)
                tensor = aug["image"].unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                all_probs.append(probs)
        return np.mean(all_probs, axis=0)

    def export_onnx(self, output_path: str = "model.onnx") -> None:
        """Export model to ONNX format."""
        dummy = torch.randn(
            1, 3, self.image_size, self.image_size, device=self.device
        )
        torch.onnx.export(
            self.model,
            dummy,
            output_path,
            opset_version=14,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )
        logger.info(f"Exported ONNX model to {output_path}")

    def export_torchscript(self, output_path: str = "model.pt") -> None:
        """Export model to TorchScript."""
        scripted = torch.jit.trace(
            self.model,
            torch.randn(1, 3, self.image_size, self.image_size, device=self.device),
        )
        scripted.save(output_path)
        logger.info(f"Exported TorchScript model to {output_path}")
