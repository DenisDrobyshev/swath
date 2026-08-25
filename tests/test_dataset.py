from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from swath.data.dataset import (
    Sample,
    SegmentationDataset,
    build_label_map,
    read_index,
    write_index,
)
from swath.data.transforms import build_eval_transform, build_train_transform
from swath.tasks import IGNORE_INDEX


def test_dataset_yields_tensors(samples, task):
    dataset = SegmentationDataset(samples, task, transform=build_train_transform(32))
    image, mask = dataset[0]
    assert isinstance(image, torch.Tensor) and image.shape == (3, 32, 32)
    assert mask.dtype == torch.long and mask.shape == (32, 32)


def test_label_map_sends_unlisted_values_to_ignore():
    table = build_label_map({1: 0, 2: 1})
    assert table[1] == 0
    assert table[2] == 1
    assert table[0] == IGNORE_INDEX
    assert table[7] == IGNORE_INDEX


def test_label_map_is_applied(samples, task):
    shifted = build_label_map({0: 2, 1: 1, 2: 0})
    dataset = SegmentationDataset(samples, task, label_map=shifted)
    _, mask = dataset.read(0)
    assert set(np.unique(mask).tolist()) <= {0, 1, 2}


def test_missing_mask_becomes_all_ignore(samples, task):
    dataset = SegmentationDataset([Sample(image=samples[0].image)], task)
    _, mask = dataset.read(0)
    assert (mask == IGNORE_INDEX).all()


def test_size_mismatch_is_reported(tmp_path: Path, samples, task):
    from PIL import Image

    small = tmp_path / "small.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(small)
    dataset = SegmentationDataset([Sample(image=samples[0].image, mask=small)], task)
    with pytest.raises(ValueError, match="mask"):
        dataset.read(0)


def test_empty_dataset_is_rejected(task):
    with pytest.raises(ValueError, match="empty"):
        SegmentationDataset([], task)


def test_pixel_counts_cover_every_class(samples, task):
    dataset = SegmentationDataset(samples, task)
    counts = dataset.pixel_counts(stride=1)
    assert counts.shape == (task.num_classes,)
    assert counts.sum() > 0
    assert (counts > 0).all()


def test_cache_returns_the_same_arrays(samples, task):
    dataset = SegmentationDataset(samples, task, cache_size=4)
    first = dataset.read(0)
    second = dataset.read(0)
    assert first[0] is second[0]


def test_index_round_trip(tmp_path: Path, samples):
    path = write_index(tmp_path / "split.json", samples)
    restored = read_index(path)
    assert [s.image for s in restored] == [s.image for s in samples]
    assert [s.mask for s in restored] == [s.mask for s in samples]


def test_eval_transform_without_crop_is_a_no_op(samples, task):
    dataset = SegmentationDataset(samples, task, transform=build_eval_transform(None))
    image, mask = dataset[0]
    assert image.shape[1:] == mask.shape
