"""Model evaluator — generates full metrics report, confusion matrix, and ROC curves."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch.amp import autocast

from models.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs inference on a DataLoader and produces a comprehensive evaluation report."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "results",
    ) -> None:
        self.model = model
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        calculator = MetricsCalculator(self.num_classes, self.class_names)
        all_probs: List[np.ndarray] = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.model(images)
            calculator.update(logits, labels)
            probs = torch.softmax(logits.cpu(), dim=1).numpy()
            all_probs.extend(probs)

        metrics = calculator.compute()
        report = calculator.classification_report()
        cm = calculator.confusion_matrix()

        logger.info("\n=== Evaluation Report ===")
        for k, v in metrics.items():
            logger.info(f"  {k:20s}: {v:.4f}")
        logger.info(f"\n{report}")

        self._plot_confusion_matrix(cm)
        self._plot_roc_curves(np.array(calculator._targets), np.array(all_probs))
        self._plot_pr_curves(np.array(calculator._targets), np.array(all_probs))
        self._save_metrics_csv(metrics)

        return metrics

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalized Confusion Matrix")
        fig.tight_layout()
        fig.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    def _plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        for i, (cls, color) in enumerate(zip(self.class_names, colors)):
            y_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, label=f"{cls} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves per Class")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(self.output_dir / "roc_curves.png", dpi=150)
        plt.close(fig)

    def _plot_pr_curves(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        for i, (cls, color) in enumerate(zip(self.class_names, colors)):
            y_bin = (y_true == i).astype(int)
            prec, rec, _ = precision_recall_curve(y_bin, y_prob[:, i])
            pr_auc = auc(rec, prec)
            ax.plot(rec, prec, color=color, label=f"{cls} (AP={pr_auc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves per Class")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(self.output_dir / "pr_curves.png", dpi=150)
        plt.close(fig)

    def _save_metrics_csv(self, metrics: Dict[str, float]) -> None:
        import pandas as pd
        pd.DataFrame([metrics]).to_csv(self.output_dir / "metrics.csv", index=False)
        logger.info(f"Metrics saved to {self.output_dir / 'metrics.csv'}")

    def compare_models(
        self,
        loaders: Dict[str, DataLoader],
        models: Dict[str, nn.Module],
    ) -> None:
        """Generate a comparison table across multiple models."""
        import pandas as pd
        rows = []
        for model_name, model in models.items():
            self.model = model.to(self.device)
            m = self.evaluate(loaders.get(model_name, list(loaders.values())[0]))
            m["model"] = model_name
            m["params"] = sum(p.numel() for p in model.parameters()) / 1e6
            rows.append(m)
        df = pd.DataFrame(rows).set_index("model")
        df.to_csv(self.output_dir / "model_comparison.csv")
        logger.info(f"\n{df.to_string()}")
