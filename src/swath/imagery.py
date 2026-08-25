"""Reading, writing and drawing raster imagery.

Everything that touches pixels but not geography lives here. Ordinary rasters
go through Pillow; anything Pillow cannot open — a sixteen-bit multispectral
GeoTIFF, say — is handed to rasterio, which is an optional dependency. Keeping
that fallback in one place means the rest of the package never has to care
which reader was used.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # aerial mosaics routinely exceed the decompression-bomb limit


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as a ``(H, W, C)`` uint8 array.

    Sixteen-bit and floating point rasters are rescaled to eight bits with a
    2–98 percentile stretch, the same contrast stretch a GIS applies when it
    first draws a scene. Without it a Sentinel-2 tile, whose reflectance values
    occupy a narrow part of the range, comes out almost black.
    """
    path = Path(path)
    array = _read_any(path)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.dtype != np.uint8:
        array = _percentile_stretch(array)
    return np.ascontiguousarray(array)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a single-band label map as a ``(H, W)`` uint8 array."""
    path = Path(path)
    array = _read_any(path)
    if array.ndim == 3:
        if array.shape[2] != 1:
            raise ValueError(
                f"{path.name}: expected a single-band label map, got {array.shape[2]} bands"
            )
        array = array[:, :, 0]
    if array.dtype != np.uint8:
        if int(array.max(initial=0)) > 255:
            raise ValueError(f"{path.name}: label values above 255 are not supported")
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array)


def _read_any(path: Path) -> np.ndarray:
    if not path.is_file():
        # Otherwise the caller gets whichever error the second reader raises,
        # which talks about drivers rather than about a missing file.
        raise FileNotFoundError(f"{path} does not exist")
    try:
        with Image.open(path) as handle:
            return np.asarray(handle)
    except Exception:
        return _read_with_rasterio(path)


def _read_with_rasterio(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as error:  # pragma: no cover - depends on the installed extras
        raise RuntimeError(
            f"cannot read {path.name} with Pillow; install the geo extra "
            f"(pip install 'swath[geo]') to read it with rasterio"
        ) from error
    with rasterio.open(path) as dataset:
        return dataset.read().transpose(1, 2, 0)


def _percentile_stretch(array: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Rescale each band to uint8 using its own percentile range."""
    values = array.astype(np.float32)
    output = np.empty(values.shape, dtype=np.uint8)
    for band in range(values.shape[2]):
        channel = values[..., band]
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            output[..., band] = 0
            continue
        lower, upper = np.percentile(finite, [low, high])
        if upper <= lower:
            lower, upper = float(finite.min()), float(finite.max())
        if upper <= lower:
            output[..., band] = 0
            continue
        scaled = (channel - lower) / (upper - lower)
        output[..., band] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return output


def colorize(mask: np.ndarray, palette: Sequence[Sequence[int]]) -> np.ndarray:
    """Map a label map to RGB using a palette.

    Labels outside the palette — the ignore value among them — are drawn black.
    """
    lookup = np.zeros((256, 3), dtype=np.uint8)
    for index, color in enumerate(palette):
        lookup[index] = color
    return lookup[mask]


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    palette: Sequence[Sequence[int]],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a colourised mask over an image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    rgb = image[:, :, :3]
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    colored = colorize(mask, palette)
    blended = rgb.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_png(path: str | Path, array: np.ndarray) -> None:
    """Write an array to PNG, accepting ``(H, W)`` or ``(H, W, 3)``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    Image.fromarray(array).save(path)


def class_pixel_counts(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Count pixels per class, ignoring labels outside ``range(num_classes)``."""
    flat = mask.reshape(-1)
    valid = flat < num_classes
    return np.bincount(flat[valid], minlength=num_classes).astype(np.int64)
