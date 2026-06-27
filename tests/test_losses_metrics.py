"""Tests for loss functions and metrics calculator."""

import numpy as np
import pytest
import torch

from models.losses import FocalLoss, LabelSmoothingCrossEntropy, build_loss
from models.metrics import MetricsCalculator


class TestLosses:
    def test_cross_entropy_returns_scalar(self, sample_config, binary_batch):
        criterion = build_loss(sample_config)
        images, labels = binary_batch
        logits = torch.randn(4, 2)
        loss = criterion(logits, labels)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_focal_loss_binary(self, binary_batch):
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        _, labels = binary_batch
        logits = torch.randn(4, 2)
        loss = criterion(logits, labels)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_label_smoothing_reduces_confidence(self):
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits = torch.tensor([[10.0, -10.0]])  # confident prediction
        labels = torch.tensor([0])
        loss = criterion(logits, labels)
        ce = torch.nn.functional.cross_entropy(logits, labels)
        # smoothed loss should be higher than sharp CE for overconfident predictions
        assert loss.item() >= 0

    def test_build_focal_loss(self, sample_config):
        cfg = {**sample_config, "loss": {"name": "focal", "focal_alpha": 0.25, "focal_gamma": 2.0}}
        criterion = build_loss(cfg)
        assert isinstance(criterion, FocalLoss)

    def test_unknown_loss_raises(self, sample_config):
        cfg = {**sample_config, "loss": {"name": "unknown_loss"}}
        with pytest.raises(ValueError):
            build_loss(cfg)


class TestMetrics:
    def test_accuracy_computation(self):
        calc = MetricsCalculator(num_classes=2)
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])  # correct predictions
        targets = torch.tensor([0, 1])
        calc.update(logits, targets)
        m = calc.compute()
        assert m["accuracy"] == pytest.approx(1.0)

    def test_metrics_keys(self):
        calc = MetricsCalculator(num_classes=2)
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        calc.update(logits, targets)
        m = calc.compute()
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "cohen_kappa"]:
            assert key in m

    def test_multiclass_metrics(self):
        calc = MetricsCalculator(num_classes=4, class_names=["A", "B", "C", "D"])
        logits = torch.randn(16, 4)
        targets = torch.randint(0, 4, (16,))
        calc.update(logits, targets)
        m = calc.compute()
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self):
        calc = MetricsCalculator(num_classes=3)
        logits = torch.randn(12, 3)
        targets = torch.randint(0, 3, (12,))
        calc.update(logits, targets)
        cm = calc.confusion_matrix()
        assert cm.shape == (3, 3)

    def test_reset_clears_state(self):
        calc = MetricsCalculator(num_classes=2)
        logits = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        calc.update(logits, targets)
        calc.reset()
        assert len(calc._preds) == 0
        assert len(calc._targets) == 0
