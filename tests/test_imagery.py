from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from swath.imagery import (
    class_pixel_counts,
    colorize,
    overlay,
    read_image,
    read_mask,
    save_png,
)


def test_read_image_returns_channels_last(tiles: Path):
    array = read_image(tiles / "images" / "tile_00.png")
    assert array.ndim == 3 and array.shape[2] == 3
    assert array.dtype == np.uint8


def test_greyscale_gains_a_channel_axis(tmp_path: Path):
    path = tmp_path / "grey.png"
    Image.fromarray(np.full((16, 16), 128, dtype=np.uint8)).save(path)
    assert read_image(path).shape == (16, 16, 1)


def test_read_mask_returns_two_dimensions(tiles: Path):
    mask = read_mask(tiles / "masks" / "tile_00.png")
    assert mask.ndim == 2
    assert mask.dtype == np.uint8


def test_sixteen_bit_input_is_stretched(tmp_path: Path):
    path = tmp_path / "wide.png"
    values = np.linspace(1000, 5000, 32 * 32).reshape(32, 32).astype(np.uint16)
    Image.fromarray(values).save(path)
    array = read_image(path)
    assert array.dtype == np.uint8
    assert array.max() == 255
    assert array.min() == 0


def test_colorize_uses_the_palette():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rgb = colorize(mask, ((10, 20, 30), (200, 100, 50)))
    assert rgb.shape == (2, 2, 3)
    assert tuple(rgb[0, 0]) == (10, 20, 30)
    assert tuple(rgb[0, 1]) == (200, 100, 50)


def test_labels_outside_the_palette_are_black():
    rgb = colorize(np.array([[255]], dtype=np.uint8), ((10, 20, 30),))
    assert tuple(rgb[0, 0]) == (0, 0, 0)


def test_overlay_blends_towards_the_class_colour():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.uint8)
    blended = overlay(image, mask, ((0, 0, 0), (200, 200, 200)), alpha=0.5)
    assert blended[0, 0].tolist() == [100, 100, 100]


def test_overlay_alpha_is_validated():
    with pytest.raises(ValueError, match="alpha"):
        overlay(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.uint8), ((0, 0, 0),), alpha=2)


def test_save_png_round_trip(tmp_path: Path):
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    save_png(tmp_path / "nested" / "mask.png", mask)
    assert np.array_equal(read_mask(tmp_path / "nested" / "mask.png"), mask)


def test_pixel_counts_ignore_out_of_range_labels():
    mask = np.array([[0, 1], [1, 255]], dtype=np.uint8)
    counts = class_pixel_counts(mask, num_classes=2)
    assert counts.tolist() == [1, 2]


def test_missing_file_says_so(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_image(tmp_path / "absent.tif")
