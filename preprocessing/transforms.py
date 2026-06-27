"""Transform utilities and helpers for medical image preprocessing."""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(
    image_size: int = 224,
    mean: list = None,
    std: list = None,
    split: str = "train",
) -> A.Compose:
    """Lightweight transform builder without full augmentation."""
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement to a BGR or grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if len(image.shape) == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_image(
    path: str,
    image_size: Optional[int] = None,
    apply_clahe_enhancement: bool = False,
) -> np.ndarray:
    """Load image as RGB numpy array."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if apply_clahe_enhancement:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bgr = apply_clahe(bgr)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        image = cv2.resize(image, (image_size, image_size))
    return image


def denormalize(
    tensor: torch.Tensor,
    mean: list = None,
    std: list = None,
) -> np.ndarray:
    """Reverse ImageNet normalization for visualization."""
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    t = tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.permute(1, 2, 0).numpy()
    return np.clip(t * 255, 0, 255).astype(np.uint8)


def resize_with_aspect(
    image: np.ndarray, target_size: int = 224
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize while preserving aspect ratio, pad to square."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return canvas, (pad_h, pad_w)
