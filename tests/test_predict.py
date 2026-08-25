from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from swath.predict import blend_weights, predict_logits, predict_mask, window_positions
from swath.tasks import Task


class ConstantModel(nn.Module):
    """Returns the same logits everywhere, so blending can be checked exactly."""

    size_divisor = 1

    def __init__(self, num_classes: int, winner: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.winner = winner
        self.parameter = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], self.num_classes, x.shape[2], x.shape[3])
        logits[:, self.winner] = 10.0
        return logits + self.parameter

    def to(self, *args, **kwargs):  # keep the CPU-only stub simple
        return self

    def eval(self):
        return self


class EdgeArtifactModel(nn.Module):
    """Right in the middle of a tile, wrong within ``margin`` px of its border.

    This is the failure that makes naive tiling visible: a pixel near the edge
    of a tile is judged with half its context missing, and the model gets it
    wrong there. Stitched without blending, those errors print a grid over the
    output at every tile boundary.
    """

    size_divisor = 1

    def __init__(self, margin: int = 8) -> None:
        super().__init__()
        self.margin = margin
        self.parameter = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        border = torch.zeros(height, width, dtype=torch.bool)
        margin = self.margin
        border[:margin, :] = border[height - margin :, :] = True
        border[:, :margin] = border[:, width - margin :] = True

        logits = torch.zeros(batch, 2, height, width)
        logits[:, 1] = torch.where(border, 0.0, 6.0)
        logits[:, 0] = torch.where(border, 6.0, 0.0)
        return logits + self.parameter

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


@pytest.fixture
def two_class_task() -> Task:
    return Task(
        name="two",
        title="Two",
        description="",
        classes=("a", "b"),
        palette=((0, 0, 0), (255, 255, 255)),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )


def test_window_positions_cover_the_edge():
    positions = window_positions(100, 64, 32)
    assert positions[0] == 0
    assert positions[-1] == 36
    assert all(p + 64 <= 100 for p in positions)


def test_window_positions_handle_a_tile_larger_than_the_image():
    assert window_positions(40, 64, 32) == [0]


def test_blend_weights_taper_to_the_edge():
    weights = blend_weights(32, taper=8)
    assert weights.shape == (32, 32)
    assert weights[16, 16] == pytest.approx(1.0)
    assert (weights > 0).all()
    # Both edges must fall away, and the ramp must be monotone on each side.
    profile = weights[16]
    assert profile[0] < profile[4] < profile[8]
    assert profile[-1] < profile[-5] < profile[-9]
    assert profile[0] == pytest.approx(profile[-1])


def test_prediction_covers_the_whole_image(two_class_task):
    image = np.zeros((150, 210, 3), dtype=np.uint8)
    mask = predict_mask(
        ConstantModel(2, winner=1), image, two_class_task, tile=64, overlap=16, device="cpu"
    )
    assert mask.shape == (150, 210)
    assert (mask == 1).all()


def test_image_smaller_than_the_tile_is_padded(two_class_task):
    image = np.zeros((30, 20, 3), dtype=np.uint8)
    mask = predict_mask(
        ConstantModel(2, winner=0), image, two_class_task, tile=64, overlap=16, device="cpu"
    )
    assert mask.shape == (30, 20)


def test_probabilities_are_normalised(two_class_task):
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    probabilities = predict_logits(
        ConstantModel(2, winner=1), image, two_class_task, tile=64, overlap=32, device="cpu"
    )
    assert probabilities.shape == (2, 80, 80)
    assert np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-4)


def _interior_error_rate(two_class_task, overlap: int) -> float:
    """Fraction of interior pixels that inherit a tile-border mistake."""
    image = np.zeros((192, 192, 3), dtype=np.uint8)
    mask = predict_mask(
        EdgeArtifactModel(margin=8),
        image,
        two_class_task,
        tile=64,
        overlap=overlap,
        device="cpu",
    )
    interior = mask[24:-24, 24:-24]
    return float((interior == 0).mean())


def test_blending_suppresses_the_tile_border_artefact(two_class_task):
    """A model that is wrong at its tile borders must not print a grid.

    Tiled without overlap, EdgeArtifactModel stamps its margin onto the output
    at every tile boundary and nearly half the image comes back wrong. With a
    tapered overlap each interior pixel is dominated by the window that holds it
    near the centre, and the artefact all but disappears — what survives is a
    scattering of pixels where two windows carry exactly equal weight and the
    tie falls to the lower class index.
    """
    with_overlap = _interior_error_rate(two_class_task, overlap=32)
    without_overlap = _interior_error_rate(two_class_task, overlap=0)

    assert without_overlap > 0.2
    assert with_overlap < 0.01
    assert with_overlap < without_overlap / 20


def test_overlap_must_be_smaller_than_the_tile(two_class_task):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="overlap"):
        predict_mask(ConstantModel(2, 0), image, two_class_task, tile=64, overlap=64, device="cpu")


def test_tta_averages_without_changing_a_constant_prediction(two_class_task):
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    mask = predict_mask(
        ConstantModel(2, winner=1),
        image,
        two_class_task,
        tile=64,
        overlap=16,
        device="cpu",
        tta=True,
    )
    assert (mask == 1).all()


def test_confidence_is_returned_when_asked(two_class_task):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask, confidence = predict_mask(
        ConstantModel(2, winner=0),
        image,
        two_class_task,
        tile=64,
        overlap=0,
        device="cpu",
        return_confidence=True,
    )
    assert mask.shape == confidence.shape
    assert confidence.min() > 0.9
