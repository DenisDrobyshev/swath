"""Inference over rasters that do not fit on the GPU.

An aerial scene is routinely tens of thousands of pixels on a side, so inference
runs as a sliding window. Cutting a raster into tiles and stitching the argmax
back together leaves visible seams: a building split across two tiles is judged
twice from two different contexts, and the join shows. The fix used here is to
overlap the windows and blend their class *probabilities* — softmax is taken per
window, before the sum, so a confident window outweighs an uncertain one — with a
raised-cosine weight that falls to zero at the tile edge, so every pixel is
dominated by the window that saw the most context around it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swath.data.transforms import normalize
from swath.models import UNet
from swath.tasks import Task


def select_device(preference: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cuda`` or ``cpu`` to a concrete device."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def accumulator_device(
    device: torch.device, num_classes: int, height: int, width: int, budget: float = 0.25
) -> torch.device:
    """Decide where to keep the blending accumulator.

    The accumulator holds one float per class per pixel, which for a 64 megapixel
    scene and seven classes is 1.8 GB — more than the model and its activations
    have to spare on an 8 GB card. When it would take more than ``budget`` of the
    device memory it goes to host memory instead: the extra transfer costs a
    fraction of a second, and the alternative is running out of memory on exactly
    the large rasters this predictor exists to handle.
    """
    if device.type != "cuda":
        return device
    required = num_classes * height * width * 4 + height * width * 4
    total = torch.cuda.get_device_properties(device).total_memory
    return device if required <= budget * total else torch.device("cpu")


def window_positions(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering ``length`` with the last window flush to the edge."""
    if tile >= length:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def blend_weights(tile: int, taper: int) -> np.ndarray:
    """A raised-cosine window that is flat in the middle and zero at the border.

    ``taper`` controls how far the ramp reaches in from each edge. A tiny floor
    is kept so that a pixel is never divided by a zero weight, which is what
    would happen in the corner of a raster covered by a single window.
    """
    taper = int(max(1, min(taper, tile // 2)))
    rise = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, taper + 2)[1:-1])
    ramp = np.ones(tile, dtype=np.float32)
    ramp[:taper] = rise
    ramp[-taper:] = rise[::-1]
    weights = np.outer(ramp, ramp)
    return np.maximum(weights, 1e-3)


_TTA_TRANSFORMS = (
    (0, False),
    (0, True),
    (2, False),
    (2, True),
)
"""Quarter turns and horizontal flips used for test-time augmentation."""


def _apply_tta(batch: torch.Tensor, turns: int, flip: bool) -> torch.Tensor:
    if turns:
        batch = torch.rot90(batch, turns, dims=(2, 3))
    if flip:
        batch = torch.flip(batch, dims=(3,))
    return batch


def _undo_tta(batch: torch.Tensor, turns: int, flip: bool) -> torch.Tensor:
    if flip:
        batch = torch.flip(batch, dims=(3,))
    if turns:
        batch = torch.rot90(batch, -turns, dims=(2, 3))
    return batch


@torch.no_grad()
def predict_logits(
    model: UNet,
    image: np.ndarray,
    task: Task,
    *,
    tile: int = 512,
    overlap: int = 128,
    batch_size: int = 4,
    device: torch.device | str = "auto",
    tta: bool = False,
    progress: bool = False,
) -> np.ndarray:
    """Run the model over a whole image and return blended per-class scores.

    Returns a ``(num_classes, H, W)`` float32 array of softmax probabilities.
    """
    if image.ndim != 3:
        raise ValueError(f"expected an (H, W, C) image, got shape {image.shape}")

    device = select_device(device) if isinstance(device, str) else device
    model = model.to(device).eval()

    # Skip connections only line up when the tile is a multiple of the divisor,
    # so a requested size is rounded down. The overlap is checked afterwards:
    # rounding 100 down to 96 can leave an overlap that no longer fits inside it,
    # and a non-positive stride loops forever.
    divisor = model.size_divisor
    requested_tile = tile
    if tile % divisor:
        tile = max(divisor, (tile // divisor) * divisor)
    if overlap >= tile:
        rounded = (
            f", rounded down from {requested_tile} to a multiple of {divisor}"
            if tile != requested_tile
            else ""
        )
        raise ValueError(
            f"overlap {overlap} must be smaller than the tile size {tile}{rounded}"
        )

    height, width = image.shape[:2]
    pad_h = max(0, tile - height)
    pad_w = max(0, tile - width)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    padded_height, padded_width = image.shape[:2]

    stride = tile - overlap
    rows = window_positions(padded_height, tile, stride)
    columns = window_positions(padded_width, tile, stride)

    weights = blend_weights(tile, taper=max(1, overlap // 2))
    weight_tensor = torch.from_numpy(weights)

    store = accumulator_device(device, task.num_classes, padded_height, padded_width)
    accumulator = torch.zeros(
        (task.num_classes, padded_height, padded_width), dtype=torch.float32, device=store
    )
    normaliser = torch.zeros((padded_height, padded_width), dtype=torch.float32, device=store)
    weight_tensor = weight_tensor.to(store)

    windows = [(row, column) for row in rows for column in columns]
    iterator: Iterator[list[tuple[int, int]]] = _batched(windows, batch_size)
    total_batches = (len(windows) + batch_size - 1) // batch_size
    if progress:
        iterator = _with_progress(iterator, total_batches, "predicting")

    use_amp = device.type == "cuda"
    for batch in iterator:
        crops = np.stack(
            [
                normalize(image[row : row + tile, column : column + tile], task.mean, task.std)
                for row, column in batch
            ]
        )
        tensor = torch.from_numpy(crops).to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if tta:
                probabilities = None
                for turns, flip in _TTA_TRANSFORMS:
                    logits = model(_apply_tta(tensor, turns, flip))
                    logits = _undo_tta(logits, turns, flip)
                    step = logits.float().softmax(dim=1)
                    probabilities = step if probabilities is None else probabilities + step
                probabilities = probabilities / len(_TTA_TRANSFORMS)
            else:
                probabilities = model(tensor).float().softmax(dim=1)

        probabilities = probabilities.float().to(store)
        for index, (row, column) in enumerate(batch):
            accumulator[:, row : row + tile, column : column + tile] += (
                probabilities[index] * weight_tensor
            )
            normaliser[row : row + tile, column : column + tile] += weight_tensor

    accumulator /= normaliser.unsqueeze(0)
    result = accumulator[:, :height, :width].cpu().numpy()
    return result


def predict_mask(
    model: UNet,
    image: np.ndarray,
    task: Task,
    *,
    return_confidence: bool = False,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Predict a label map, and optionally the winning probability per pixel."""
    probabilities = predict_logits(model, image, task, **kwargs)
    mask = probabilities.argmax(axis=0).astype(np.uint8)
    if return_confidence:
        return mask, probabilities.max(axis=0)
    return mask


def predict_file(
    model: UNet,
    path: str | Path,
    task: Task,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Predict for a raster on disk.

    Returns the image as read, the predicted label map, and the georeferencing
    when the file carried any.
    """
    from swath.geo import read_georeferenced

    image, reference = read_georeferenced(path)
    if image.shape[2] != task.in_channels:
        image = _fit_channels(image, task.in_channels)
    mask = predict_mask(model, image, task, **kwargs)
    return image, mask, reference


def _fit_channels(image: np.ndarray, expected: int) -> np.ndarray:
    channels = image.shape[2]
    if channels == expected:
        return image
    if channels == 1 and expected > 1:
        return np.repeat(image, expected, axis=2)
    if channels > expected:
        return image[:, :, :expected]
    raise ValueError(f"image has {channels} bands but the model expects {expected}")


def _batched(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _with_progress(iterator: Iterator[Any], total: int, description: str) -> Iterator[Any]:
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        yield from iterator
        return
    yield from tqdm(iterator, total=total, desc=description, unit="batch")
