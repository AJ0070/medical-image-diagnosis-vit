"""PyTorch Dataset and DataModule for medical imaging tasks."""

import os
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from preprocessing.augmentation import build_augmentation_pipeline
from preprocessing.transforms import load_image

logger = logging.getLogger(__name__)


class MedicalImageDataset(Dataset):
    """Unified PyTorch Dataset for all medical imaging tasks."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
        apply_clahe: bool = False,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or []
        self.apply_clahe = apply_clahe

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = load_image(
            self.image_paths[idx],
            apply_clahe_enhancement=self.apply_clahe,
        )
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        label = self.labels[idx]
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced datasets."""
        counts = np.bincount(self.labels)
        total = len(self.labels)
        weights = total / (len(counts) * counts.astype(float))
        return torch.tensor(weights, dtype=torch.float32)


def _scan_directory(root: str) -> Tuple[List[str], List[int], List[str]]:
    """Scan a directory tree where each subfolder is a class."""
    root = Path(root)
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for cls in class_names:
        cls_dir = root / cls
        for fp in cls_dir.iterdir():
            if fp.suffix.lower() in valid_exts:
                paths.append(str(fp))
                labels.append(class_to_idx[cls])
    return paths, labels, class_names


def _load_from_csv(
    csv_path: str,
    image_col: str = "image_id",
    label_col: str = "label",
    image_dir: str = "",
) -> Tuple[List[str], List[int]]:
    """Load paths and labels from a CSV annotation file."""
    df = pd.read_csv(csv_path)
    paths = [os.path.join(image_dir, str(p)) for p in df[image_col].tolist()]
    labels = df[label_col].tolist()
    return paths, labels


class DataModule:
    """Data loading orchestrator: split, sample, and build DataLoaders."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.dataset_cfg = cfg.get("dataset", {})
        self.data_cfg = cfg.get("data", {})
        self.train_cfg = cfg.get("training", {})
        self.aug_cfg = cfg.get("augmentation", {})

        self.num_classes: int = self.dataset_cfg.get("num_classes", 2)
        self.class_names: List[str] = self.dataset_cfg.get("classes", [])
        self.data_root: str = self.dataset_cfg.get("path", "datasets")
        self.apply_clahe: bool = self.aug_cfg.get("clahe", False)
        # pin_memory only helps when a CUDA GPU is available
        self._pin_memory: bool = (
            self.train_cfg.get("pin_memory", True) and torch.cuda.is_available()
        )

        self.train_dataset: Optional[MedicalImageDataset] = None
        self.val_dataset: Optional[MedicalImageDataset] = None
        self.test_dataset: Optional[MedicalImageDataset] = None

    def setup(self, csv_path: Optional[str] = None) -> None:
        """Load and split data, build datasets.

        Supports two layouts:
        1. Pre-split: root/train/<class>/, root/val/<class>/, root/test/<class>/
        2. Flat:      root/<class>/  (DataModule performs the split)
        """
        train_transform = build_augmentation_pipeline(self.cfg, split="train")
        eval_transform = build_augmentation_pipeline(self.cfg, split="val")

        root = Path(self.data_root)
        if self._is_presplit(root):
            # Use the existing train/val/test directories
            train_paths, train_labels, class_names = _scan_directory(str(root / "train"))
            val_dir = root / "val" if (root / "val").exists() else root / "valid"
            val_paths, val_labels, _ = _scan_directory(str(val_dir))
            test_dir = root / "test"
            if test_dir.exists():
                test_paths, test_labels, _ = _scan_directory(str(test_dir))
            else:
                test_paths, test_labels = val_paths, val_labels
            if not self.class_names:
                self.class_names = class_names
            logger.info("Detected pre-split dataset layout (train/val/test folders)")
        elif csv_path and os.path.exists(csv_path):
            all_paths, all_labels = _load_from_csv(csv_path, image_dir=self.data_root)
            train_paths, temp_paths, train_labels, temp_labels = self._split(all_paths, all_labels)
            val_paths, test_paths, val_labels, test_labels = self._split_val_test(
                temp_paths, temp_labels
            )
        else:
            all_paths, all_labels, class_names = _scan_directory(self.data_root)
            if not self.class_names:
                self.class_names = class_names
            train_paths, temp_paths, train_labels, temp_labels = self._split(all_paths, all_labels)
            val_paths, test_paths, val_labels, test_labels = self._split_val_test(
                temp_paths, temp_labels
            )

        self.train_dataset = MedicalImageDataset(
            train_paths, train_labels, train_transform, self.class_names, self.apply_clahe
        )
        self.val_dataset = MedicalImageDataset(
            val_paths, val_labels, eval_transform, self.class_names, self.apply_clahe
        )
        self.test_dataset = MedicalImageDataset(
            test_paths, test_labels, eval_transform, self.class_names, self.apply_clahe
        )

        logger.info(
            f"Dataset split — train: {len(train_paths)}, "
            f"val: {len(val_paths)}, test: {len(test_paths)}"
        )

    @staticmethod
    def _is_presplit(root: Path) -> bool:
        return (root / "train").is_dir()

    def _split(self, paths, labels):
        train_ratio = self.data_cfg.get("train_split", 0.7)
        return train_test_split(
            paths, labels, test_size=1 - train_ratio, stratify=labels, random_state=42
        )

    def _split_val_test(self, paths, labels):
        val_ratio = self.data_cfg.get("val_split", 0.15)
        train_ratio = self.data_cfg.get("train_split", 0.7)
        relative_val = val_ratio / (1 - train_ratio)
        return train_test_split(
            paths, labels, test_size=1 - relative_val, stratify=labels, random_state=42
        )

    def _make_sampler(self, dataset: MedicalImageDataset) -> Optional[WeightedRandomSampler]:
        """Balanced sampler for imbalanced classes."""
        class_weights = dataset.get_class_weights()
        sample_weights = [class_weights[l].item() for l in dataset.labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def train_dataloader(self, use_sampler: bool = True) -> DataLoader:
        sampler = self._make_sampler(self.train_dataset) if use_sampler else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_cfg.get("batch_size", 32),
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.train_cfg.get("num_workers", 4),
            pin_memory=self._pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_cfg.get("batch_size", 32) * 2,
            shuffle=False,
            num_workers=self.train_cfg.get("num_workers", 4),
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.train_cfg.get("batch_size", 32) * 2,
            shuffle=False,
            num_workers=self.train_cfg.get("num_workers", 4),
            pin_memory=self._pin_memory,
        )
