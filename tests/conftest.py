"""Shared pytest fixtures for the test suite."""

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def sample_config():
    return {
        "project": {"name": "test-project", "seed": 42, "checkpoint_dir": "checkpoints", "log_dir": "logs", "output_dir": "results"},
        "dataset": {"name": "test", "path": "datasets/test", "num_classes": 2, "classes": ["Normal", "Pneumonia"]},
        "model": {"name": "efficientnet_b0", "pretrained": False, "dropout": 0.1, "image_size": 64, "freeze_epochs": 0},
        "training": {"epochs": 2, "batch_size": 4, "num_workers": 0, "pin_memory": False, "mixed_precision": False, "gradient_accumulation_steps": 1, "early_stopping_patience": 3, "gradient_clip_norm": 1.0},
        "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-4},
        "scheduler": {"name": "cosine", "warmup_epochs": 0, "min_lr": 1e-6},
        "loss": {"name": "cross_entropy", "label_smoothing": 0.0},
        "data": {"train_split": 0.7, "val_split": 0.15, "test_split": 0.15, "image_size": 64, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "augmentation": {"enabled": True, "horizontal_flip": True, "vertical_flip": False, "rotation": 10, "brightness": 0.1, "contrast": 0.1, "gaussian_noise": 0.0, "clahe": False, "random_crop": False},
        "logging": {"tensorboard": {"enabled": False}, "wandb": {"enabled": False}},
        "inference": {"tta_enabled": False, "tta_steps": 3, "batch_size": 4},
    }


@pytest.fixture
def sample_image():
    """Random 224×224 RGB image as numpy array."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor():
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def binary_batch():
    images = torch.randn(4, 3, 64, 64)
    labels = torch.randint(0, 2, (4,))
    return images, labels


@pytest.fixture
def multiclass_batch():
    images = torch.randn(4, 3, 64, 64)
    labels = torch.randint(0, 4, (4,))
    return images, labels
