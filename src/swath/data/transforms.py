"""Augmentations for image and mask pairs.

Every transform takes and returns a ``(image, mask)`` pair, where the image is
``(H, W, C)`` and the mask is ``(H, W)``. Geometry is applied to both; anything
that changes colour is applied to the image alone. Masks are always resampled
with nearest neighbour, since interpolating class indices invents classes that
do not exist.

Aerial imagery has no canonical orientation, which is why the full dihedral
group is used here (flips plus quarter turns) rather than the horizontal flip
that is standard for photographs.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from PIL import Image

from swath.tasks import IGNORE_INDEX

Pair = tuple[np.ndarray, np.ndarray]


class Transform:
    """Base class: a callable that maps an image and mask pair to another."""

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Pair:  # pragma: no cover
        raise NotImplementedError


class RandomTransform(Transform):
    """Base class for transforms that draw randomness.

    The generator is created lazily and re-created whenever the process id
    changes. Without that, a generator built in the parent process is pickled
    into every DataLoader worker with the same internal state, and four workers
    then apply the identical sequence of crops and flips — augmentation that
    looks random per sample but repeats across the batch.
    """

    def __init__(self) -> None:
        self._rng: np.random.Generator | None = None
        self._pid: int | None = None

    @property
    def rng(self) -> np.random.Generator:
        pid = os.getpid()
        if self._rng is None or self._pid != pid:
            self._rng = np.random.default_rng()
            self._pid = pid
        return self._rng


class Compose(Transform):
    """Apply transforms in order."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Pair:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class RandomScale(RandomTransform):
    """Resize by a random factor, keeping the aspect ratio."""

    def __init__(self, low: float = 0.75, high: float = 1.25, probability: float = 1.0) -> None:
        super().__init__()
        if low <= 0 or high < low:
            raise ValueError("expected 0 < low <= high")
        self.low, self.high = low, high
        self.probability = probability

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Pair:
        if self.rng.random() > self.probability:
            return image, mask
        scale = float(self.rng.uniform(self.low, self.high))
        height, width = mask.shape
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))
        mask = np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST))
        return image, mask


class RandomCrop(RandomTransform):
    """Crop a random window, padding first when the image is smaller."""

    def __init__(self, size: int, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.size = size
        self.ignore_index = ignore_index

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Pair:
        image, mask = pad_to(image, mask, self.size, self.ignore_index)
        height, width = mask.shape
        top = int(self.rng.integers(0, height - self.size + 1))
        left = int(self.rng.integers(0, width - self.size + 1))
        return (
            image[top : top + self.size, left : left + self.size],
            mask[top : top + self.size, left : left + self.size],
        )


class CenterCrop(Transform):
    """Crop the central window, padding first when the image is smaller."""

    def __init__(self, size: int, ignore_index: int = IGNORE_INDEX) -> None:
        self.size = size
        self.ignore_index = ignore_index

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Pair:
        image, mask = pad_to(image, mask, self.size, self.ignore_index)
        height, width = mask.shape
        top = (height - self.size) // 2
        left = (width - self.size) // 2
        return (
            image[top : top + self.size, left : left + self.size],
            mask[top : top + self.size, left : left + self.size],
        )


class RandomDihedral(RandomTransform):
    """One of the eight flips and quarter turns of the square."""

    def __init__(self, probability: float = 1.0) -> None:
        super().__init__()
        self.probability = probability

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Pair:
        if self.rng.random() > self.probability:
            return image, mask
        turns = int(self.rng.integers(0, 4))
        if turns:
            image = np.rot90(image, turns, axes=(0, 1))
            mask = np.rot90(mask, turns, axes=(0, 1))
        if self.rng.random() < 0.5:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class ColorJitter(RandomTransform):
    """Random brightness, contrast and saturation, applied to the image only.

    Kept mild on purpose. Strong colour augmentation helps on photographs, but
    on land cover the colour *is* much of the signal: bleaching a forest towards
    grey moves it into the barren class.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.probability = probability

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Pair:
        if self.rng.random() > self.probability:
            return image, mask
        values = image.astype(np.float32)
        if self.brightness:
            values *= 1.0 + float(self.rng.uniform(-self.brightness, self.brightness))
        if self.contrast:
            mean = values.mean()
            factor = 1.0 + float(self.rng.uniform(-self.contrast, self.contrast))
            values = (values - mean) * factor + mean
        if self.saturation and values.shape[-1] >= 3:
            grey = values[..., :3].mean(axis=-1, keepdims=True)
            factor = 1.0 + float(self.rng.uniform(-self.saturation, self.saturation))
            values[..., :3] = (values[..., :3] - grey) * factor + grey
        return np.clip(values, 0, 255).astype(image.dtype), mask


def pad_to(
    image: np.ndarray, mask: np.ndarray, size: int, ignore_index: int = IGNORE_INDEX
) -> Pair:
    """Pad the bottom and right edges so both sides are at least ``size``.

    Padded image pixels are zero and padded mask pixels are the ignore label, so
    the padding never contributes to the loss.
    """
    height, width = mask.shape
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    if not pad_h and not pad_w:
        return image, mask
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=ignore_index)
    return image, mask


def normalize(
    image: np.ndarray, mean: Sequence[float], std: Sequence[float], scale: float = 255.0
) -> np.ndarray:
    """Scale to ``[0, 1]`` and standardise, returning float32 in ``(C, H, W)``."""
    values = image.astype(np.float32) / scale
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    if values.shape[-1] != mean_array.size:
        raise ValueError(
            f"image has {values.shape[-1]} channels but mean/std describe {mean_array.size}"
        )
    values = (values - mean_array) / std_array
    return np.ascontiguousarray(values.transpose(2, 0, 1))


def build_train_transform(crop_size: int = 512, scale: tuple[float, float] = (0.75, 1.25)):
    """The default training pipeline: random scale, crop, dihedral, colour."""
    return Compose(
        [
            RandomScale(scale[0], scale[1]),
            RandomCrop(crop_size),
            RandomDihedral(),
            ColorJitter(),
        ]
    )


def build_eval_transform(crop_size: int | None = None):
    """The default evaluation pipeline: a centre crop, or nothing at all."""
    return Compose([CenterCrop(crop_size)] if crop_size else [])
