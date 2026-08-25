"""Evaluation at full resolution.

Validation during training runs on crops, because it has to be cheap enough to
run every other epoch. That is fine for choosing between checkpoints and wrong
for reporting a number: a crop is not the tile a user will actually submit, and
the sliding-window path — the one that stitches a large raster back together —
is never exercised. This module runs the real thing: whole tiles, through the
same predictor the CLI and the service use, into one confusion matrix.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swath.data.dataset import Sample
from swath.imagery import read_image, read_mask
from swath.metrics import ConfusionMatrix, MetricResult
from swath.models import UNet
from swath.predict import predict_mask, select_device
from swath.tasks import Task


@dataclass
class EvaluationReport:
    """Metrics plus enough context to know how they were produced."""

    metrics: MetricResult
    samples: int
    tile: int
    overlap: int
    tta: bool
    task: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "samples": self.samples,
            "tile": self.tile,
            "overlap": self.overlap,
            "tta": self.tta,
            "overall_accuracy": round(self.metrics.overall_accuracy, 5),
            "mean_iou": round(self.metrics.mean_iou, 5),
            "mean_f1": round(self.metrics.mean_f1, 5),
            "per_class_iou": {k: round(v, 5) for k, v in self.metrics.per_class_iou.items()},
            "per_class_f1": {k: round(v, 5) for k, v in self.metrics.per_class_f1.items()},
        }

    def summary(self) -> str:
        return (
            f"{self.samples} tiles, tile {self.tile} overlap {self.overlap}"
            f"{' with TTA' if self.tta else ''}\n{self.metrics.table()}"
        )


@torch.no_grad()
def evaluate(
    model: UNet,
    samples: Sequence[Sample],
    task: Task,
    *,
    label_map: np.ndarray | None = None,
    tile: int = 512,
    overlap: int = 128,
    batch_size: int = 4,
    device: str | torch.device = "auto",
    tta: bool = False,
    progress: bool = True,
) -> EvaluationReport:
    """Segment every sample at full resolution and accumulate the metrics."""
    labelled = [sample for sample in samples if sample.mask is not None]
    if not labelled:
        raise ValueError("none of the samples carry a label map to score against")

    resolved = select_device(device) if isinstance(device, str) else device
    model = model.to(resolved).eval()
    matrix = ConfusionMatrix(task.num_classes, device=resolved)

    iterator: Any = labelled
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(labelled, desc="evaluating", unit="tile")
        except ImportError:  # pragma: no cover
            pass

    for sample in iterator:
        image = read_image(sample.image)
        truth = read_mask(sample.mask)
        if label_map is not None:
            truth = label_map[truth]

        prediction = predict_mask(
            model,
            image,
            task,
            tile=tile,
            overlap=overlap,
            batch_size=batch_size,
            device=resolved,
            tta=tta,
        )
        matrix.update(
            torch.from_numpy(truth.astype(np.int64)),
            torch.from_numpy(prediction.astype(np.int64)),
        )

    return EvaluationReport(
        metrics=matrix.compute(task.classes),
        samples=len(labelled),
        tile=tile,
        overlap=overlap,
        tta=tta,
        task=task.name,
    )


def write_report(path: str | Path, report: EvaluationReport) -> Path:
    """Write a report as JSON next to the checkpoint that produced it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return path
