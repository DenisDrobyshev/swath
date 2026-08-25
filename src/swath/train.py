"""The training loop.

Nothing exotic: AdamW, a cosine schedule with a short warm-up, mixed precision
on CUDA, and gradient clipping. The parts worth pointing at are the ones that
decide whether the numbers in the README can be trusted — validation runs on
full tiles rather than random crops, the confusion matrix is accumulated across
the whole split, and the checkpoint kept as *best* is chosen on mean IoU rather
than on loss, because loss and mIoU disagree exactly where the rare classes are.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from swath.checkpoints import save_checkpoint
from swath.data.dataset import SegmentationDataset
from swath.losses import SegmentationLoss, class_weights_from_frequency
from swath.metrics import ConfusionMatrix, MetricResult
from swath.models import build_model
from swath.predict import select_device
from swath.tasks import Task


@dataclass
class TrainConfig:
    """Everything a run needs, in one serialisable place."""

    output_dir: Path = Path("runs/default")
    epochs: int = 30
    batch_size: int = 8
    crop_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 1
    min_learning_rate_factor: float = 0.02
    grad_clip: float = 1.0
    base_channels: int = 48
    depth: int = 4
    blocks_per_stage: int = 2
    norm: str = "batch"
    dropout: float = 0.1
    dice_weight: float = 0.5
    label_smoothing: float = 0.02
    class_weight_mode: str = "sqrt_inverse"
    num_workers: int = 4
    seed: int = 0
    amp: bool = True
    device: str = "auto"
    val_interval: int = 1
    val_crop_size: int | None = 1024
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> str:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return json.dumps(payload, indent=2)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate_at(step: int, total_steps: int, warmup_steps: int, floor: float) -> float:
    """Linear warm-up into a cosine decay, as a multiplier on the base rate."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return floor + (1 - floor) * cosine


class Trainer:
    """Owns the model, the optimiser and the bookkeeping for one run."""

    def __init__(
        self,
        task: Task,
        train_dataset: SegmentationDataset,
        val_dataset: SegmentationDataset | None,
        config: TrainConfig,
    ) -> None:
        self.task = task
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = select_device(config.device)

        seed_everything(config.seed)

        self.model = build_model(
            in_channels=task.in_channels,
            num_classes=task.num_classes,
            base_channels=config.base_channels,
            depth=config.depth,
            blocks_per_stage=config.blocks_per_stage,
            norm=config.norm,
            dropout=config.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.use_amp = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.criterion = SegmentationLoss(
            weight=self._class_weights(),
            dice_weight=config.dice_weight,
            label_smoothing=config.label_smoothing,
        ).to(self.device)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, float]] = []
        self.best_score = -1.0
        self.start_epoch = 0

    def _class_weights(self) -> torch.Tensor | None:
        if self.config.class_weight_mode == "none":
            return None
        counts = torch.from_numpy(self.train_dataset.pixel_counts(limit=200, stride=8))
        weights = class_weights_from_frequency(counts, mode=self.config.class_weight_mode)
        return weights.to(self.device)

    def loaders(self) -> tuple[DataLoader, DataLoader | None]:
        workers = self.config.num_workers
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=workers > 0,
        )
        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=max(1, self.config.batch_size // 2),
                shuffle=False,
                num_workers=max(0, workers // 2),
                pin_memory=self.device.type == "cuda",
            )
        return train_loader, val_loader

    def train_one_epoch(self, loader: DataLoader, epoch: int, total_epochs: int) -> float:
        self.model.train()
        steps_per_epoch = len(loader)
        total_steps = steps_per_epoch * total_epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs

        running = 0.0
        seen = 0
        started = time.time()

        for index, (images, targets) in enumerate(loader):
            global_step = epoch * steps_per_epoch + index
            factor = learning_rate_at(
                global_step, total_steps, warmup_steps, self.config.min_learning_rate_factor
            )
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.learning_rate * factor

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            if self.config.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += float(loss.detach()) * images.size(0)
            seen += images.size(0)

            if index % 20 == 0 or index == steps_per_epoch - 1:
                elapsed = time.time() - started
                rate = (index + 1) / max(elapsed, 1e-6)
                print(
                    f"  epoch {epoch + 1}/{total_epochs} "
                    f"step {index + 1}/{steps_per_epoch} "
                    f"loss {running / max(seen, 1):.4f} "
                    f"lr {self.config.learning_rate * factor:.2e} "
                    f"{rate:.2f} it/s",
                    flush=True,
                )

        return running / max(seen, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, MetricResult]:
        self.model.eval()
        matrix = ConfusionMatrix(self.task.num_classes, device=self.device)
        running = 0.0
        seen = 0

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            matrix.update(targets, logits.argmax(dim=1))
            running += float(loss.detach()) * images.size(0)
            seen += images.size(0)

        return running / max(seen, 1), matrix.compute(self.task.classes)

    def fit(self) -> dict[str, Any]:
        train_loader, val_loader = self.loaders()
        (self.output_dir / "config.json").write_text(self.config.as_json(), encoding="utf-8")

        print(
            f"training {self.task.name} on {len(self.train_dataset)} tiles, "
            f"validating on {len(self.val_dataset) if self.val_dataset else 0}, "
            f"{self.model.num_parameters() / 1e6:.2f}M parameters, device {self.device}",
            flush=True,
        )

        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_started = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch, self.config.epochs)

            record: dict[str, float] = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 5),
                "seconds": round(time.time() - epoch_started, 1),
            }

            should_validate = val_loader is not None and (
                (epoch + 1) % self.config.val_interval == 0 or epoch + 1 == self.config.epochs
            )
            if should_validate:
                val_loss, metrics = self.evaluate(val_loader)
                record.update(
                    val_loss=round(val_loss, 5),
                    val_miou=round(metrics.mean_iou, 5),
                    val_mf1=round(metrics.mean_f1, 5),
                    val_oa=round(metrics.overall_accuracy, 5),
                )
                for name, value in metrics.per_class_iou.items():
                    record[f"iou_{name}"] = round(value, 5)
                print(f"  validation: {metrics.summary()}", flush=True)

                if metrics.mean_iou > self.best_score:
                    self.best_score = metrics.mean_iou
                    save_checkpoint(
                        self.output_dir / "best.pt",
                        self.model,
                        self.task,
                        epoch=epoch + 1,
                        metrics={
                            "mean_iou": metrics.mean_iou,
                            "mean_f1": metrics.mean_f1,
                            "overall_accuracy": metrics.overall_accuracy,
                        },
                        history=self.history,
                        notes=self.config.notes,
                    )
                    (self.output_dir / "metrics.txt").write_text(
                        metrics.table() + "\n", encoding="utf-8"
                    )
                    print(f"  new best mIoU {metrics.mean_iou:.4f}, checkpoint saved", flush=True)

            self.history.append(record)
            self._write_history()
            save_checkpoint(
                self.output_dir / "last.pt",
                self.model,
                self.task,
                epoch=epoch + 1,
                metrics={"mean_iou": self.best_score},
                history=self.history,
                notes=self.config.notes,
            )

        return {"best_mean_iou": self.best_score, "history": self.history}

    def _write_history(self) -> None:
        if not self.history:
            return
        columns: list[str] = []
        for record in self.history:
            for key in record:
                if key not in columns:
                    columns.append(key)
        path = self.output_dir / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.history)
