"""Comprehensive evaluation metrics for medical image classification."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)


class MetricsCalculator:
    """Accumulates predictions and computes all evaluation metrics at epoch end."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        self._preds: List[int] = []
        self._targets: List[int] = []
        self._probs: List[np.ndarray] = []
        self._loss_sum: float = 0.0
        self._n: int = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None,
    ) -> None:
        probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()
        preds = probs.argmax(axis=1).tolist()
        self._preds.extend(preds)
        self._targets.extend(targets.cpu().tolist())
        self._probs.extend(probs)
        if loss is not None:
            self._loss_sum += loss * len(targets)
            self._n += len(targets)

    def compute(self) -> Dict[str, float]:
        y_true = np.array(self._targets)
        y_pred = np.array(self._preds)
        y_prob = np.array(self._probs)

        avg = "binary" if self.num_classes == 2 else "macro"
        multi_class = "raise" if self.num_classes == 2 else "ovr"

        metrics: Dict[str, float] = {}

        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        metrics["specificity"] = self._specificity(y_true, y_pred)
        metrics["sensitivity"] = metrics["recall"]
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        metrics["matthews_cc"] = float(matthews_corrcoef(y_true, y_pred))

        try:
            if self.num_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_prob[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class=multi_class, average="macro")
                )
                metrics["pr_auc"] = float(
                    np.mean(
                        [average_precision_score((y_true == c).astype(int), y_prob[:, c])
                         for c in range(self.num_classes)]
                    )
                )
        except ValueError:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

        if self._n > 0:
            metrics["loss"] = self._loss_sum / self._n

        return metrics

    def _specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Macro-averaged specificity (true negative rate)."""
        specs = []
        for c in range(self.num_classes):
            tn = int(((y_true != c) & (y_pred != c)).sum())
            fp = int(((y_true != c) & (y_pred == c)).sum())
            specs.append(tn / (tn + fp + 1e-8))
        return float(np.mean(specs))

    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self._targets, self._preds)

    def classification_report(self) -> str:
        return classification_report(
            self._targets,
            self._preds,
            target_names=self.class_names,
            zero_division=0,
        )

    def per_class_accuracy(self) -> Dict[str, float]:
        cm = self.confusion_matrix()
        per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
        return {self.class_names[i]: float(per_class[i]) for i in range(self.num_classes)}
