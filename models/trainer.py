"""Training engine with AMP, gradient accumulation, early stopping, and W&B/TB logging."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = float("-inf") if mode == "max" else float("inf")
        self.stop = False

    def __call__(self, score: float) -> bool:
        improved = (
            score > self.best + self.min_delta
            if self.mode == "max"
            else score < self.best - self.min_delta
        )
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class Trainer:
    """Full training loop with mixed precision, logging, checkpointing, and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        num_classes: int,
        class_names: Optional[list] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.num_classes = num_classes
        self.class_names = class_names

        self.train_cfg = cfg.get("training", {})
        self.opt_cfg = cfg.get("optimizer", {})
        self.sch_cfg = cfg.get("scheduler", {})
        self.project_cfg = cfg.get("project", {})

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")

        self.epochs = self.train_cfg.get("epochs", 50)
        self.grad_accum = self.train_cfg.get("gradient_accumulation_steps", 1)
        self.clip_norm = self.train_cfg.get("gradient_clip_norm", 1.0)
        self.amp = self.train_cfg.get("mixed_precision", True) and self.device.type == "cuda"
        self.freeze_epochs = cfg.get("model", {}).get("freeze_epochs", 0)

        self.scaler = GradScaler(device_type=self.device.type, enabled=self.amp)
        self.optimizer = self._build_optimizer()
        self.early_stopping = EarlyStopping(
            patience=self.train_cfg.get("early_stopping_patience", 10)
        )

        # Checkpoint dir
        ckpt_dir = Path(self.project_cfg.get("checkpoint_dir", "checkpoints"))
        project_name = self.project_cfg.get("name", "medvit")
        self.ckpt_dir = ckpt_dir / project_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        log_dir = Path(self.project_cfg.get("log_dir", "logs")) / project_name
        tb_cfg = cfg.get("logging", {}).get("tensorboard", {})
        self.writer = SummaryWriter(log_dir=str(log_dir)) if tb_cfg.get("enabled", True) else None

        # W&B
        self.wandb_run = None
        wb_cfg = cfg.get("logging", {}).get("wandb", {})
        if wb_cfg.get("enabled", False):
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wb_cfg.get("project", "medvit"),
                    entity=wb_cfg.get("entity"),
                    config=cfg,
                    name=project_name,
                )
            except ImportError:
                logger.warning("wandb not installed — skipping W&B logging")

        self.best_score = float("-inf")
        self.current_epoch = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.opt_cfg.get("name", "adamw").lower()
        lr = float(self.opt_cfg.get("lr", 1e-4))
        wd = float(self.opt_cfg.get("weight_decay", 1e-4))
        params = self.model.parameters()
        if name == "adam":
            return Adam(params, lr=lr, weight_decay=wd)
        elif name == "adamw":
            return AdamW(params, lr=lr, weight_decay=wd)
        elif name == "sgd":
            return SGD(params, lr=lr, momentum=self.opt_cfg.get("momentum", 0.9), weight_decay=wd)
        elif name == "rmsprop":
            return RMSprop(params, lr=lr, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self, train_loader: DataLoader):
        name = self.sch_cfg.get("name", "cosine").lower()
        warmup = self.sch_cfg.get("warmup_epochs", 0)

        if name == "cosine":
            main = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - warmup,
                eta_min=float(self.sch_cfg.get("min_lr", 1e-6)),
            )
        elif name == "step":
            main = StepLR(
                self.optimizer,
                step_size=self.sch_cfg.get("step_size", 10),
                gamma=self.sch_cfg.get("gamma", 0.1),
            )
        elif name == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="max", patience=5, factor=0.5)
        elif name == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=float(self.opt_cfg.get("lr", 1e-4)),
                epochs=self.epochs,
                steps_per_epoch=len(train_loader),
            )
        else:
            main = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        if warmup > 0:
            warmup_sched = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup)
            return SequentialLR(self.optimizer, [warmup_sched, main], milestones=[warmup])
        return main

    def _unfreeze_if_needed(self) -> None:
        if self.freeze_epochs > 0 and self.current_epoch == self.freeze_epochs:
            raw = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            raw.unfreeze_backbone()

    def train_epoch(
        self, loader: DataLoader, criterion: nn.Module, metrics: MetricsCalculator
    ) -> Dict[str, float]:
        self.model.train()
        metrics.reset()
        self.optimizer.zero_grad()

        for step, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(images)
                loss = criterion(logits, labels) / self.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            metrics.update(logits.detach(), labels.detach(), loss.item() * self.grad_accum)

        return metrics.compute()

    @torch.no_grad()
    def eval_epoch(
        self, loader: DataLoader, criterion: nn.Module, metrics: MetricsCalculator
    ) -> Dict[str, float]:
        self.model.eval()
        metrics.reset()
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(images)
                loss = criterion(logits, labels)
            metrics.update(logits, labels, loss.item())
        return metrics.compute()

    def _log(self, prefix: str, metrics: dict, epoch: int) -> None:
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, epoch)
        if self.wandb_run:
            self.wandb_run.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=epoch)

    def _save_checkpoint(self, epoch: int, score: float, is_best: bool) -> None:
        raw = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        state = {
            "epoch": epoch,
            "score": score,
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cfg": self.cfg,
        }
        torch.save(state, self.ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best_model.pth")
            logger.info(f"  Saved best model — score: {score:.4f}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> None:
        scheduler = self._build_scheduler(train_loader)
        train_metrics = MetricsCalculator(self.num_classes, self.class_names)
        val_metrics = MetricsCalculator(self.num_classes, self.class_names)

        # Optionally freeze backbone at start
        if self.freeze_epochs > 0:
            raw = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            raw.freeze_backbone()

        # Resume from checkpoint
        start_epoch = self._maybe_resume()

        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            self._unfreeze_if_needed()
            t0 = time.time()

            train_result = self.train_epoch(train_loader, criterion, train_metrics)
            val_result = self.eval_epoch(val_loader, criterion, val_metrics)

            # Scheduler step
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_result.get("f1", 0))
            elif not isinstance(scheduler, OneCycleLR):
                scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            score = val_result.get("f1", val_result.get("accuracy", 0))

            self._log("train", train_result, epoch)
            self._log("val", val_result, epoch)
            if self.writer:
                self.writer.add_scalar("lr", lr, epoch)

            is_best = score > self.best_score
            if is_best:
                self.best_score = score
            self._save_checkpoint(epoch, score, is_best)

            logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"loss={train_result.get('loss', 0):.4f} | "
                f"val_acc={val_result.get('accuracy', 0):.4f} | "
                f"val_f1={val_result.get('f1', 0):.4f} | "
                f"val_auc={val_result.get('roc_auc', 0):.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            if self.early_stopping(score):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()

    def _maybe_resume(self) -> int:
        resume = self.train_cfg.get("resume_from")
        if resume and os.path.exists(resume):
            state = torch.load(resume, map_location=self.device)
            raw = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            raw.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            start = state.get("epoch", 0) + 1
            logger.info(f"Resumed from {resume} — starting at epoch {start}")
            return start
        return 0
