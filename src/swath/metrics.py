"""Segmentation metrics built on a streaming confusion matrix.

Averaging per-batch IoU is a common and wrong shortcut: a batch that happens to
contain no pixels of a rare class still contributes a score for it. Accumulating
one confusion matrix over the whole epoch and deriving the metrics at the end
avoids that, and costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from swath.tasks import IGNORE_INDEX


@dataclass
class MetricResult:
    """Metrics derived from a confusion matrix."""

    overall_accuracy: float
    mean_iou: float
    mean_f1: float
    per_class_iou: dict[str, float]
    per_class_f1: dict[str, float]

    def summary(self) -> str:
        return (
            f"OA {self.overall_accuracy:.4f}  "
            f"mIoU {self.mean_iou:.4f}  "
            f"mF1 {self.mean_f1:.4f}"
        )

    def table(self) -> str:
        width = max((len(name) for name in self.per_class_iou), default=5)
        lines = [f"{'class'.ljust(width)}  {'IoU':>7}  {'F1':>7}"]
        for name, iou in self.per_class_iou.items():
            lines.append(f"{name.ljust(width)}  {iou:7.4f}  {self.per_class_f1[name]:7.4f}")
        lines.append(f"{'mean'.ljust(width)}  {self.mean_iou:7.4f}  {self.mean_f1:7.4f}")
        return "\n".join(lines)


class ConfusionMatrix:
    """Accumulates predictions into a ``num_classes x num_classes`` matrix.

    Rows are ground truth, columns are predictions. Pixels labelled
    :data:`~swath.tasks.IGNORE_INDEX` are dropped before counting.
    """

    def __init__(self, num_classes: int, device: torch.device | str = "cpu") -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    def reset(self) -> None:
        self.matrix.zero_()

    @torch.no_grad()
    def update(self, target: torch.Tensor, prediction: torch.Tensor) -> None:
        """Add one batch of integer label maps to the matrix."""
        if target.shape != prediction.shape:
            raise ValueError(
                f"target shape {tuple(target.shape)} does not match "
                f"prediction shape {tuple(prediction.shape)}"
            )
        target = target.reshape(-1).to(self.matrix.device)
        prediction = prediction.reshape(-1).to(self.matrix.device)
        valid = target != IGNORE_INDEX
        target, prediction = target[valid], prediction[valid]
        if target.numel() == 0:
            return
        indices = target.to(torch.int64) * self.num_classes + prediction.to(torch.int64)
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self, class_names: list[str] | tuple[str, ...] | None = None) -> MetricResult:
        """Derive accuracy, IoU and F1 from the accumulated counts.

        Classes that never appear in the ground truth and are never predicted
        are left out of the means, so an unused class cannot drag mIoU down.
        """
        matrix = self.matrix.to(torch.float64)
        true_positive = matrix.diag()
        actual = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        union = actual + predicted - true_positive

        present = union > 0
        iou = torch.where(present, true_positive / union.clamp(min=1), torch.zeros_like(union))
        denominator = actual + predicted
        f1 = torch.where(
            present, 2 * true_positive / denominator.clamp(min=1), torch.zeros_like(union)
        )

        total = matrix.sum()
        overall_accuracy = float(true_positive.sum() / total) if total > 0 else 0.0
        mean_iou = float(iou[present].mean()) if bool(present.any()) else 0.0
        mean_f1 = float(f1[present].mean()) if bool(present.any()) else 0.0

        names = list(class_names) if class_names else [str(i) for i in range(self.num_classes)]
        if len(names) != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} class names, got {len(names)}"
            )

        return MetricResult(
            overall_accuracy=overall_accuracy,
            mean_iou=mean_iou,
            mean_f1=mean_f1,
            per_class_iou={name: float(value) for name, value in zip(names, iou, strict=True)},
            per_class_f1={name: float(value) for name, value in zip(names, f1, strict=True)},
        )
