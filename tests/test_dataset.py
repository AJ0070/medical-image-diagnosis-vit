"""Tests for dataset loading, transforms, and augmentation pipeline."""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from preprocessing.augmentation import build_augmentation_pipeline, build_tta_pipeline
from preprocessing.transforms import build_transforms, load_image, denormalize, apply_clahe
from preprocessing.dataset import MedicalImageDataset


class TestTransforms:
    def test_build_train_transforms(self, sample_config):
        t = build_augmentation_pipeline(sample_config, split="train")
        assert t is not None

    def test_build_eval_transforms(self, sample_config):
        t = build_augmentation_pipeline(sample_config, split="val")
        assert t is not None

    def test_transform_output_shape(self, sample_config, sample_image):
        t = build_augmentation_pipeline(sample_config, split="val")
        out = t(image=sample_image)
        tensor = out["image"]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 3  # channels first

    def test_denormalize_returns_uint8(self):
        tensor = torch.randn(3, 64, 64)
        out = denormalize(tensor)
        assert out.dtype == np.uint8
        assert out.shape == (64, 64, 3)

    def test_clahe_on_grayscale(self):
        gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        out = apply_clahe(gray)
        assert out.shape == (64, 64)

    def test_tta_returns_multiple_transforms(self, sample_config):
        data = sample_config.get("data", {})
        pipelines = build_tta_pipeline(64, data.get("mean"), data.get("std"))
        assert len(pipelines) == 5


class TestDataset:
    def _create_dummy_dataset(self, tmpdir: str, num_classes: int = 2, n_per_class: int = 5):
        """Create a directory of dummy images organized by class folder."""
        class_names = [f"class_{i}" for i in range(num_classes)]
        paths, labels = [], []
        for i, cls in enumerate(class_names):
            cls_dir = Path(tmpdir) / cls
            cls_dir.mkdir()
            for j in range(n_per_class):
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                path = str(cls_dir / f"img_{j}.png")
                cv2.imwrite(path, img)
                paths.append(path)
                labels.append(i)
        return paths, labels, class_names

    def test_dataset_length(self, tmp_path):
        paths, labels, _ = self._create_dummy_dataset(str(tmp_path))
        ds = MedicalImageDataset(paths, labels)
        assert len(ds) == 10

    def test_dataset_getitem_no_transform(self, tmp_path):
        paths, labels, _ = self._create_dummy_dataset(str(tmp_path))
        ds = MedicalImageDataset(paths, labels)
        img, label = ds[0]
        assert isinstance(label, int)
        assert isinstance(img, np.ndarray)

    def test_dataset_getitem_with_transform(self, tmp_path, sample_config):
        paths, labels, _ = self._create_dummy_dataset(str(tmp_path))
        t = build_augmentation_pipeline(sample_config, split="val")
        ds = MedicalImageDataset(paths, labels, transform=t)
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 3

    def test_class_weights_shape(self, tmp_path):
        paths, labels, _ = self._create_dummy_dataset(str(tmp_path), num_classes=3, n_per_class=4)
        ds = MedicalImageDataset(paths, labels)
        weights = ds.get_class_weights()
        assert weights.shape == (3,)
        assert (weights > 0).all()
