from __future__ import annotations

import numpy as np

from swath.data.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    RandomCrop,
    RandomDihedral,
    RandomScale,
    build_train_transform,
    normalize,
    pad_to,
)
from swath.tasks import IGNORE_INDEX


def _pair(size=32):
    image = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[: size // 2] = 1
    return image, mask


def test_random_crop_returns_the_requested_size():
    image, mask = _pair(64)
    cropped_image, cropped_mask = RandomCrop(32)(image, mask)
    assert cropped_image.shape == (32, 32, 3)
    assert cropped_mask.shape == (32, 32)


def test_crop_pads_when_the_tile_is_too_small():
    image, mask = _pair(16)
    cropped_image, cropped_mask = CenterCrop(32)(image, mask)
    assert cropped_image.shape == (32, 32, 3)
    assert IGNORE_INDEX in np.unique(cropped_mask)


def test_padding_uses_the_ignore_label():
    image, mask = _pair(8)
    padded_image, padded_mask = pad_to(image, mask, 12)
    assert padded_image.shape == (12, 12, 3)
    assert padded_mask[-1, -1] == IGNORE_INDEX
    assert padded_image[-1, -1].sum() == 0


def test_dihedral_preserves_the_label_set():
    image, mask = _pair()
    before = set(np.unique(mask).tolist())
    for _ in range(8):
        image, mask = RandomDihedral()(image, mask)
    assert set(np.unique(mask).tolist()) == before
    assert image.flags["C_CONTIGUOUS"]


def test_scale_keeps_image_and_mask_aligned():
    image, mask = _pair(64)
    scaled_image, scaled_mask = RandomScale(0.5, 0.5)(image, mask)
    assert scaled_image.shape[:2] == scaled_mask.shape
    # Nearest-neighbour resampling must not invent a third class.
    assert set(np.unique(scaled_mask).tolist()) <= {0, 1}


def test_color_jitter_leaves_the_mask_alone():
    image, mask = _pair()
    jittered_image, jittered_mask = ColorJitter(probability=1.0)(image, mask)
    assert np.array_equal(mask, jittered_mask)
    assert jittered_image.dtype == np.uint8


def test_normalize_produces_channel_first_float():
    image, _ = _pair()
    array = normalize(image, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    assert array.shape == (3, 32, 32)
    assert array.dtype == np.float32


def test_default_pipeline_yields_the_crop_size():
    image, mask = _pair(96)
    pipeline = build_train_transform(crop_size=64)
    for _ in range(5):
        out_image, out_mask = pipeline(image, mask)
        assert out_image.shape == (64, 64, 3)
        assert out_mask.shape == (64, 64)


def test_compose_is_ordered():
    image, mask = _pair(64)
    pipeline = Compose([CenterCrop(48), CenterCrop(32)])
    out_image, _ = pipeline(image, mask)
    assert out_image.shape == (32, 32, 3)
