"""Albumentations-based augmentation pipelines for medical imaging."""

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
        transforms = _build_train_pipeline(aug, image_size, mean, std)
    else:
        transforms = _build_eval_pipeline(image_size, mean, std)

    return transforms


def _build_train_pipeline(aug: dict, image_size: int, mean: list, std: list) -> A.Compose:
    ops = [A.Resize(image_size, image_size)]

    if aug.get("random_crop", False):
        ops.append(A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)))

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
        ops.append(A.GaussNoise(var_limit=(noise * 255, noise * 2 * 255), p=0.3))

    ops.extend(
        [
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.5),
                    A.GridDistortion(p=0.5),
                ],
                p=0.2,
            ),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return A.Compose(ops)


def _build_eval_pipeline(image_size: int, mean: list, std: list) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_tta_pipeline(image_size: int, mean: list, std: list) -> list[A.Compose]:
    """Test-time augmentation pipelines."""
    base = [A.Resize(image_size, image_size), A.Normalize(mean=mean, std=std), ToTensorV2()]
    return [
        A.Compose(base),
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0)] + base[1:]),
        A.Compose([A.Resize(image_size, image_size), A.VerticalFlip(p=1.0)] + base[1:]),
        A.Compose(
            [A.Resize(image_size, image_size), A.Rotate(limit=90, p=1.0)] + base[1:]
        ),
        A.Compose(
            [
                A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
                A.CenterCrop(image_size, image_size),
            ]
            + base[1:]
        ),
    ]
