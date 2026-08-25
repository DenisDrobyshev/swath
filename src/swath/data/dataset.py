"""A dataset over pairs of image and label-map files.

The dataset is deliberately format-agnostic: it takes an explicit list of file
pairs rather than crawling a directory layout of its own invention. Anything
that knows how a particular corpus is arranged on disk — see
:mod:`swath.data.loveda` — produces that list, and everything downstream works
the same way.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from swath.data.transforms import Transform, normalize
from swath.imagery import class_pixel_counts, read_image, read_mask
from swath.tasks import IGNORE_INDEX, Task


@dataclass(frozen=True)
class Sample:
    """One image, and the label map that goes with it when there is one."""

    image: Path
    mask: Path | None = None

    def as_dict(self, root: Path | None = None) -> dict[str, str | None]:
        image = self.image.relative_to(root) if root else self.image
        mask = (self.mask.relative_to(root) if root else self.mask) if self.mask else None
        return {"image": image.as_posix(), "mask": mask.as_posix() if mask else None}

    @classmethod
    def from_dict(cls, payload: dict[str, str | None], root: Path | None = None) -> Sample:
        image = Path(payload["image"])
        mask = Path(payload["mask"]) if payload.get("mask") else None
        if root:
            image = root / image
            mask = root / mask if mask else None
        return cls(image=image, mask=mask)


def build_label_map(mapping: dict[int, int], ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Build a 256-entry lookup table that remaps raw label values.

    Values absent from ``mapping`` become ``ignore_index``. Datasets tend to
    reserve a raw value for "no data" — remapping through a table drops it into
    the ignore label in a single vectorised pass.
    """
    table = np.full(256, ignore_index, dtype=np.uint8)
    for raw, target in mapping.items():
        if not 0 <= raw <= 255:
            raise ValueError(f"raw label {raw} is outside the byte range")
        table[raw] = target
    return table


class SegmentationDataset(Dataset):
    """Yields ``(image, mask)`` tensors ready for a segmentation model.

    Args:
        samples: Image and label-map pairs.
        task: Supplies the class count and the normalisation statistics.
        transform: Augmentation pipeline applied to the raw arrays.
        label_map: Optional 256-entry lookup applied to every label map.
        cache_size: Number of decoded samples to keep in memory. Useful for the
            small validation split; leave at zero for training.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        task: Task,
        transform: Transform | None = None,
        label_map: np.ndarray | None = None,
        cache_size: int = 0,
    ) -> None:
        if not samples:
            raise ValueError("dataset is empty")
        self.samples = list(samples)
        self.task = task
        self.transform = transform
        self.label_map = label_map
        self.cache_size = cache_size
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def read(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read one raw pair, applying the label map but no augmentation."""
        if index in self._cache:
            return self._cache[index]

        sample = self.samples[index]
        image = read_image(sample.image)
        if image.shape[2] != self.task.in_channels:
            image = _fit_channels(image, self.task.in_channels, sample.image)

        if sample.mask is None:
            mask = np.full(image.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        else:
            mask = read_mask(sample.mask)
            if self.label_map is not None:
                mask = self.label_map[mask]
            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"{sample.image.name}: image is {image.shape[:2]} but "
                    f"the mask is {mask.shape}"
                )

        if self.cache_size and len(self._cache) < self.cache_size:
            self._cache[index] = (image, mask)
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.read(index)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        tensor = torch.from_numpy(normalize(image, self.task.mean, self.task.std))
        labels = torch.from_numpy(np.array(mask, dtype=np.int64, copy=True))
        return tensor, labels

    def pixel_counts(self, limit: int | None = None, stride: int = 4) -> np.ndarray:
        """Count pixels per class over the split, for class-weighted losses.

        Reads every ``stride``-th pixel of at most ``limit`` samples: the class
        balance of a corpus is stable enough that a subsample settles it, and a
        full pass over a few thousand large tiles is not worth the minutes.
        """
        total = np.zeros(self.task.num_classes, dtype=np.int64)
        indices = range(len(self.samples)) if limit is None else range(min(limit, len(self)))
        for index in indices:
            _, mask = self.read(index)
            total += class_pixel_counts(mask[::stride, ::stride], self.task.num_classes)
        return total


def _fit_channels(image: np.ndarray, expected: int, path: Path) -> np.ndarray:
    """Reconcile a band count that does not match the task.

    A greyscale scene is repeated across the expected bands and an RGBA PNG has
    its alpha dropped; anything else is a genuine mismatch and is reported as one.
    """
    channels = image.shape[2]
    if channels == 1 and expected > 1:
        return np.repeat(image, expected, axis=2)
    if channels > expected:
        return image[:, :, :expected]
    raise ValueError(
        f"{path.name}: image has {channels} bands but the task expects {expected}"
    )


def write_index(path: str | Path, samples: Iterable[Sample], root: Path | None = None) -> Path:
    """Write a split to a JSON index so a training run is reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.as_dict(root) for sample in samples]
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def read_index(path: str | Path, root: Path | None = None) -> list[Sample]:
    """Read a split written by :func:`write_index`."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Sample.from_dict(entry, root) for entry in payload]
