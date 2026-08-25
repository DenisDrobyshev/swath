"""Shared fixtures.

Everything the tests need is synthesised here: a handful of tiny tiles with a
deterministic pattern, and a checkpoint trained for a few steps on them. That
keeps the suite fast and, more importantly, keeps it runnable in CI where the
real corpus is not available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from swath.checkpoints import save_checkpoint
from swath.models import build_model
from swath.tasks import Task


@pytest.fixture
def task() -> Task:
    return Task(
        name="test",
        title="Test task",
        description="Three classes over synthetic tiles.",
        classes=("background", "stripe", "blob"),
        palette=((0, 0, 0), (220, 40, 40), (40, 120, 220)),
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
    )


def synthesize(index: int, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """A tile whose classes are recoverable from colour alone."""
    rng = np.random.default_rng(index)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[:, size // 3 : size // 3 + size // 6] = 1

    yy, xx = np.mgrid[0:size, 0:size]
    centre = (size * 2 // 3, size // 2 + (index % 5) * 2)
    blob = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 < (size // 6) ** 2
    mask[blob] = 2

    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[mask == 0] = (30, 90, 40)
    image[mask == 1] = (210, 60, 50)
    image[mask == 2] = (50, 110, 210)
    noise = rng.integers(-12, 12, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image, mask


@pytest.fixture
def tiles(tmp_path: Path) -> Path:
    """A directory of eight image and mask pairs."""
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for index in range(8):
        image, mask = synthesize(index)
        Image.fromarray(image).save(images / f"tile_{index:02d}.png")
        Image.fromarray(mask).save(masks / f"tile_{index:02d}.png")
    return tmp_path


@pytest.fixture
def samples(tiles: Path):
    from swath.data.dataset import Sample

    return [
        Sample(image=tiles / "images" / f"tile_{index:02d}.png",
               mask=tiles / "masks" / f"tile_{index:02d}.png")
        for index in range(8)
    ]


@pytest.fixture
def tiny_model(task: Task):
    return build_model(
        in_channels=task.in_channels,
        num_classes=task.num_classes,
        base_channels=8,
        depth=2,
        blocks_per_stage=1,
        dropout=0.0,
    )


@pytest.fixture
def checkpoint(tmp_path: Path, tiny_model, task: Task) -> Path:
    path = tmp_path / "run" / "best.pt"
    save_checkpoint(
        path,
        tiny_model,
        task,
        epoch=3,
        metrics={"mean_iou": 0.5, "overall_accuracy": 0.9},
        notes="synthetic fixture",
    )
    return path
