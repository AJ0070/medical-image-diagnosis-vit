"""Albumentations-based augmentation pipelines for medical imaging (compatible with v2.x)."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_augmentation_pipeline(cfg: dict, split: str = "train") -> A.Compose:
    """Build augmentation pipeline from config dict."""
    aug = cfg.get("augmentation", {})
    data = cfg.get("data", {})
    image_size = data.get("image_size", 224)
    mean = data.get("mean", [0.485, 0.456, 0.406])
    std = data.get("std", [0.229, 0.224, 0.225])

    if split == "train" and aug.get("enabled", True):
        return _build_train_pipeline(aug, image_size, mean, std)
    return _build_eval_pipeline(image_size, mean, std)


def _build_train_pipeline(aug: dict, image_size: int, mean: list, std: list) -> A.Compose:
    ops = [A.Resize(height=image_size, width=image_size)]

    if aug.get("random_crop", False):
        # v2: size must be a tuple (h, w)
        ops.append(A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)))

    if aug.get("horizontal_flip", True):
        ops.append(A.HorizontalFlip(p=0.5))

    if aug.get("vertical_flip", False):
        ops.append(A.VerticalFlip(p=0.5))

    rotation = aug.get("rotation", 0)
    if rotation > 0:
        ops.append(A.Rotate(limit=rotation, p=0.7))

    brightness = aug.get("brightness", 0.0)
    contrast = aug.get("contrast", 0.0)
    if brightness > 0 or contrast > 0:
        ops.append(
            A.RandomBrightnessContrast(
                brightness_limit=brightness, contrast_limit=contrast, p=0.5
            )
        )

    if aug.get("clahe", False):
        ops.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3))

    noise = aug.get("gaussian_noise", 0.0)
    if noise > 0:
        # v2: std_range replaces var_limit
        std_low = noise
        std_high = noise * 2
        ops.append(A.GaussNoise(std_range=(std_low, std_high), p=0.3))

    ops.extend(
        [
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.4),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.5),
                    A.GridDistortion(p=0.5),
                ],
                p=0.2,
            ),
            # v2: num_holes_range / hole_height_range / hole_width_range replace max_* params
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0,
                p=0.2,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return A.Compose(ops)


def _build_eval_pipeline(image_size: int, mean: list, std: list) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_tta_pipeline(image_size: int, mean: list, std: list) -> list:
    """Test-time augmentation pipelines."""
    norm = [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return [
        A.Compose([A.Resize(height=image_size, width=image_size)] + norm),
        A.Compose([A.Resize(height=image_size, width=image_size), A.HorizontalFlip(p=1.0)] + norm),
        A.Compose([A.Resize(height=image_size, width=image_size), A.VerticalFlip(p=1.0)] + norm),
        A.Compose([A.Resize(height=image_size, width=image_size), A.Rotate(limit=90, p=1.0)] + norm),
        A.Compose(
            [
                A.Resize(height=int(image_size * 1.1), width=int(image_size * 1.1)),
                A.CenterCrop(height=image_size, width=image_size),
            ]
            + norm
        ),
    ]
