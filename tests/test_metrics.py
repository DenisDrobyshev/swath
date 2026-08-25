from __future__ import annotations

import torch

from swath.metrics import ConfusionMatrix
from swath.tasks import IGNORE_INDEX


def test_perfect_prediction_scores_one():
    matrix = ConfusionMatrix(3)
    target = torch.tensor([[0, 1, 2, 2]])
    matrix.update(target, target.clone())
    result = matrix.compute(["a", "b", "c"])
    assert result.mean_iou == 1.0
    assert result.overall_accuracy == 1.0


def test_iou_matches_hand_computation():
    # Two pixels of class 1 predicted, one of them correct; one class-1 pixel missed.
    target = torch.tensor([0, 0, 1, 1])
    prediction = torch.tensor([0, 1, 1, 0])
    matrix = ConfusionMatrix(2)
    matrix.update(target, prediction)
    result = matrix.compute(["background", "object"])

    assert result.per_class_iou["background"] == 1 / 3
    assert result.per_class_iou["object"] == 1 / 3
    assert result.overall_accuracy == 0.5


def test_ignore_index_is_excluded():
    target = torch.tensor([0, 1, IGNORE_INDEX, IGNORE_INDEX])
    prediction = torch.tensor([0, 1, 1, 0])
    matrix = ConfusionMatrix(2)
    matrix.update(target, prediction)
    result = matrix.compute(["background", "object"])

    assert result.overall_accuracy == 1.0
    assert int(matrix.matrix.sum()) == 2


def test_absent_class_does_not_drag_the_mean_down():
    # Class 2 never appears and is never predicted; it must not count as a zero.
    target = torch.tensor([0, 0, 1, 1])
    matrix = ConfusionMatrix(3)
    matrix.update(target, target.clone())
    result = matrix.compute(["a", "b", "c"])
    assert result.mean_iou == 1.0
    assert result.per_class_iou["c"] == 0.0


def test_reset_clears_counts():
    matrix = ConfusionMatrix(2)
    matrix.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    matrix.reset()
    assert int(matrix.matrix.sum()) == 0
