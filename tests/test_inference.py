"""Tests for the Predictor inference pipeline."""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPredictor:
    def _make_mock_model(self, num_classes: int = 2):
        model = MagicMock()
        model.parameters.return_value = iter([torch.tensor([1.0])])
        model.eval.return_value = model
        model.to.return_value = model
        logits = torch.zeros(1, num_classes)
        logits[0, 0] = 5.0
        model.return_value = logits
        model.__call__ = lambda self, x: logits
        return model

    def test_predict_numpy_image(self, sample_config):
        from inference.predictor import Predictor

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = iter([])
        mock_model.return_value = torch.tensor([[5.0, -5.0]])
        mock_model.__call__ = MagicMock(return_value=torch.tensor([[5.0, -5.0]]))

        predictor = Predictor(
            model=mock_model,
            cfg=sample_config,
            checkpoint_path=None,
            device=torch.device("cpu"),
        )
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image, generate_gradcam=False, generate_attention=False)
        assert result.predicted_class in (0, 1)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == 2

    def test_predict_pil_image(self, sample_config):
        from inference.predictor import Predictor

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = iter([])
        mock_model.__call__ = MagicMock(return_value=torch.tensor([[5.0, -5.0]]))

        predictor = Predictor(
            model=mock_model,
            cfg=sample_config,
            checkpoint_path=None,
            device=torch.device("cpu"),
        )
        pil_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = predictor.predict(pil_image, generate_gradcam=False, generate_attention=False)
        assert result.predicted_label in sample_config["dataset"]["classes"]

    def test_to_dict_has_required_keys(self, sample_config):
        from inference.predictor import PredictionResult

        result = PredictionResult(
            predicted_class=0,
            predicted_label="Normal",
            confidence=0.95,
            probabilities={"Normal": 0.95, "Pneumonia": 0.05},
        )
        d = result.to_dict()
        assert "predicted_class" in d
        assert "predicted_label" in d
        assert "confidence" in d
        assert "probabilities" in d
