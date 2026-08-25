from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from swath.predict import (
    accumulator_device,
    blend_weights,
    predict_logits,
    predict_mask,
    window_positions,
)
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
    with pytest.raises(ValueError, match=r"overlap 64 must be smaller than the tile size 64$"):
        predict_mask(
            ConstantModel(2, 0), image, two_class_task, tile=64, overlap=64, device="cpu"
        )


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


def test_accumulator_stays_on_cpu_for_a_cpu_model():
    cpu = torch.device("cpu")
    assert accumulator_device(cpu, 7, 100_000, 100_000) == cpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_accumulator_moves_to_host_memory_when_it_would_not_fit():
    cuda = torch.device("cuda")
    total = torch.cuda.get_device_properties(cuda).total_memory

    # A tile-sized accumulator belongs on the device.
    assert accumulator_device(cuda, 7, 512, 512).type == "cuda"

    # One sized at the whole device certainly does not.
    side = int((total / (7 * 4)) ** 0.5)
    assert accumulator_device(cuda, 7, side, side).type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_host_accumulation_gives_the_same_answer(two_class_task):
    """Where the sums are kept must not change what comes out of them.

    The model runs on the GPU either way; only the accumulator moves. Softmax
    outputs are computed identically, so the two paths must agree to float
    precision — if they did not, the fallback would silently change results on
    exactly the large rasters that trigger it.
    """
    from unittest.mock import patch

    from swath.models import build_model

    torch.manual_seed(0)
    model = build_model(num_classes=2, base_channels=8, depth=2, blocks_per_stage=1).eval()
    image = (np.random.default_rng(1).random((128, 160, 3)) * 255).astype(np.uint8)

    on_device = predict_logits(
        model, image, two_class_task, tile=64, overlap=16, device="cuda"
    )
    with patch("swath.predict.accumulator_device", return_value=torch.device("cpu")):
        on_host = predict_logits(
            model, image, two_class_task, tile=64, overlap=16, device="cuda"
        )

    assert np.allclose(on_device, on_host, atol=1e-5)
    assert on_device.argmax(axis=0).tolist() == on_host.argmax(axis=0).tolist()


def test_tile_is_rounded_to_the_model_stride(two_class_task):
    class Depth3Model(ConstantModel):
        size_divisor = 8

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    # 100 is not a multiple of 8; it becomes 96, and the run must still cover
    # the image rather than tripping over a stride that no longer fits.
    mask = predict_mask(
        Depth3Model(2, winner=1), image, two_class_task, tile=100, overlap=16, device="cpu"
    )
    assert mask.shape == (128, 128)
    assert (mask == 1).all()


def test_overlap_is_checked_against_the_rounded_tile(two_class_task):
    class Depth3Model(ConstantModel):
        size_divisor = 8

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    # 98 rounds down to 96, so an overlap of 97 stops fitting and would otherwise
    # produce a non-positive stride.
    with pytest.raises(ValueError, match="rounded down from 98 to a multiple of 8"):
        predict_mask(
            Depth3Model(2, winner=1), image, two_class_task, tile=98, overlap=97, device="cpu"
        )
