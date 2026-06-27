"""Tests for EfficientNet and ViT model builders."""

import pytest
import torch

from models.vit import build_vit
from models.efficientnet import build_efficientnet
from models import build_model


class TestEfficientNet:
    def test_output_shape_binary(self, sample_config, sample_tensor):
        model = build_efficientnet(sample_config, num_classes=2)
        out = model(sample_tensor)
        assert out.shape == (1, 2)

    def test_output_shape_multiclass(self, sample_config, sample_tensor):
        cfg = {**sample_config, "model": {**sample_config["model"], "name": "efficientnet_b0"}}
        model = build_efficientnet(cfg, num_classes=4)
        out = model(sample_tensor)
        assert out.shape == (1, 4)

    def test_freeze_unfreeze(self, sample_config):
        model = build_efficientnet(sample_config, num_classes=2)
        model.freeze_backbone()
        frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        assert frozen > 0

        model.unfreeze_backbone()
        still_frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        assert still_frozen == 0

    def test_param_count(self, sample_config):
        model = build_efficientnet(sample_config, num_classes=2)
        assert model.count_params() > 0
        assert model.count_trainable_params() > 0


class TestViT:
    def test_output_shape(self, sample_config):
        cfg = {**sample_config, "model": {**sample_config["model"], "name": "vit_b16", "image_size": 224}}
        tensor = torch.randn(1, 3, 224, 224)
        model = build_vit(cfg, num_classes=2)
        out = model(tensor)
        assert out.shape == (1, 2)

    def test_freeze(self, sample_config):
        cfg = {**sample_config, "model": {**sample_config["model"], "name": "vit_b16", "image_size": 224}}
        model = build_vit(cfg, num_classes=2)
        model.freeze_backbone()
        trainable = model.count_trainable_params()
        total = model.count_params()
        assert trainable < total


class TestModelFactory:
    def test_build_efficientnet_from_config(self, sample_config):
        model = build_model(sample_config, num_classes=2)
        assert model is not None

    def test_unknown_model_raises(self, sample_config):
        cfg = {**sample_config, "model": {**sample_config["model"], "name": "unknown_model"}}
        with pytest.raises(ValueError):
            build_model(cfg, num_classes=2)
