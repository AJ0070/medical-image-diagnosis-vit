"""Loss functions: Cross-Entropy, Weighted CE, Focal Loss, Class-Balanced Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance in binary and multi-class settings."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ClassBalancedLoss(nn.Module):
    """Class-balanced loss using effective number of samples (Cui et al. 2019)."""

    def __init__(self, samples_per_class: list, beta: float = 0.9999, gamma: float = 0.5) -> None:
        super().__init__()
        effective_num = 1.0 - torch.tensor(samples_per_class, dtype=torch.float32) ** beta
        weights = (1.0 - beta) / effective_num
        self.weights = weights / weights.sum() * len(samples_per_class)
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        if self.weight is not None:
            w = self.weight.to(inputs.device)[targets]
            loss = -(smooth_targets * log_probs).sum(dim=1) * w
        else:
            loss = -(smooth_targets * log_probs).sum(dim=1)
        return loss.mean()


def build_loss(cfg: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Build loss function from config."""
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "cross_entropy")
    smoothing = loss_cfg.get("label_smoothing", 0.0)

    if name == "cross_entropy":
        if smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
        return nn.CrossEntropyLoss(weight=class_weights)

    elif name == "weighted_cross_entropy":
        if smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
        return nn.CrossEntropyLoss(weight=class_weights)

    elif name == "focal":
        return FocalLoss(
            alpha=loss_cfg.get("focal_alpha", 0.25),
            gamma=loss_cfg.get("focal_gamma", 2.0),
        )

    raise ValueError(f"Unknown loss function: {name}")
