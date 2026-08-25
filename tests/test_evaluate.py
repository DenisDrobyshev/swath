from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from swath.data.dataset import Sample, build_label_map
from swath.evaluate import evaluate, write_report


class PerfectModel(torch.nn.Module):
    """Reads the answer off the colours, the way the synthetic tiles encode it."""

    size_divisor = 1

    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The tiles are green for background, red for the stripe, blue for the blob.
        red, green, blue = x[:, 0], x[:, 1], x[:, 2]
        logits = torch.stack([green, red, blue], dim=1) * 8.0
        return logits + self.parameter

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def test_evaluation_scores_a_perfect_model(samples, task):
    report = evaluate(
        PerfectModel(), samples, task, tile=64, overlap=16, device="cpu", progress=False
    )
    assert report.samples == len(samples)
    assert report.metrics.mean_iou > 0.9
    assert report.metrics.overall_accuracy > 0.95


def test_report_records_how_it_was_produced(samples, task):
    report = evaluate(
        PerfectModel(), samples[:2], task, tile=64, overlap=0, tta=True,
        device="cpu", progress=False,
    )
    payload = report.as_dict()
    assert payload["samples"] == 2
    assert payload["tile"] == 64
    assert payload["overlap"] == 0
    assert payload["tta"] is True
    assert set(payload["per_class_iou"]) == set(task.classes)


def test_unlabelled_samples_are_refused(samples, task):
    with pytest.raises(ValueError, match="label map"):
        evaluate(
            PerfectModel(),
            [Sample(image=samples[0].image)],
            task,
            device="cpu",
            progress=False,
        )


def test_label_map_is_applied_to_the_ground_truth(samples, task):
    # Remapping every raw label onto class 0 must make a perfect model look wrong.
    flatten = build_label_map({0: 0, 1: 0, 2: 0})
    report = evaluate(
        PerfectModel(), samples[:2], task, label_map=flatten, tile=64, overlap=0,
        device="cpu", progress=False,
    )
    assert report.metrics.overall_accuracy < 0.9


def test_write_report_round_trips(tmp_path: Path, samples, task):
    report = evaluate(
        PerfectModel(), samples[:2], task, tile=64, overlap=0, device="cpu", progress=False
    )
    path = write_report(tmp_path / "nested" / "evaluation.json", report)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mean_iou"] == pytest.approx(report.metrics.mean_iou, abs=1e-5)


def test_summary_is_human_readable(samples, task):
    report = evaluate(
        PerfectModel(), samples[:2], task, tile=64, overlap=0, device="cpu", progress=False
    )
    text = report.summary()
    assert "2 tiles" in text
    assert "stripe" in text
    assert "mean" in text


def test_evaluation_matches_a_hand_built_confusion(task):
    """A model that always answers class 0 scores exactly the background share."""

    class AlwaysBackground(PerfectModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
            logits[:, 0] = 10.0
            return logits + self.parameter

    import tempfile

    from PIL import Image

    from conftest import synthesize

    image_array, mask_array = synthesize(0)
    expected = float((mask_array == 0).mean())

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        Image.fromarray(image_array).save(root / "image.png")
        Image.fromarray(mask_array).save(root / "mask.png")
        report = evaluate(
            AlwaysBackground(),
            [Sample(image=root / "image.png", mask=root / "mask.png")],
            task,
            tile=64,
            overlap=0,
            device="cpu",
            progress=False,
        )

    assert report.metrics.overall_accuracy == pytest.approx(expected, abs=1e-6)
    assert np.isclose(report.metrics.per_class_iou["stripe"], 0.0)
