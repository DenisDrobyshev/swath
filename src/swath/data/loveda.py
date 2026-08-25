"""The LoveDA land-cover corpus.

LoveDA is 0.3 m aerial imagery over three Chinese cities, split into a rural and
an urban domain and labelled with seven land-cover classes. It is a good default
for this package: the download needs no registration, the licence is permissive,
and the two domains make the usual failure of land-cover models — a model that
learned one kind of landscape and collapses on the other — visible in the metrics
rather than hidden in them.

Reference:
    Wang et al., *LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive
    Semantic Segmentation*, NeurIPS 2021 Datasets and Benchmarks.
    https://doi.org/10.5281/zenodo.5706578 (CC BY 4.0)
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from swath.data.dataset import Sample, build_label_map

ZENODO_RECORD = "https://zenodo.org/records/5706578"
ARCHIVES = {
    "Train": f"{ZENODO_RECORD}/files/Train.zip",
    "Val": f"{ZENODO_RECORD}/files/Val.zip",
    "Test": f"{ZENODO_RECORD}/files/Test.zip",
}

DOMAINS = ("Rural", "Urban")
SPLITS = ("Train", "Val", "Test")

RAW_LABELS = {
    0: "no-data",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}
"""Label values as they appear in the distributed PNG masks."""

RAW_TO_TASK = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
"""Raw value to ``landcover`` class index. Raw 0 is no-data and becomes ignore."""


def label_map() -> np.ndarray:
    """Lookup table remapping raw LoveDA labels onto the ``landcover`` task."""
    return build_label_map(RAW_TO_TASK)


def extract(archive: str | Path, destination: str | Path) -> Path:
    """Unpack one LoveDA archive, skipping the work if it is already unpacked."""
    archive, destination = Path(archive), Path(destination)
    if not archive.is_file():
        raise FileNotFoundError(
            f"{archive} not found; download it from {ARCHIVES.get(archive.stem, ZENODO_RECORD)}"
        )
    marker = destination / archive.stem
    if marker.is_dir() and any(marker.rglob("*.png")):
        return marker
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    if not marker.is_dir():
        raise RuntimeError(f"{archive.name} did not unpack into {marker}")
    return marker


def discover(root: str | Path, split: str, domains: tuple[str, ...] = DOMAINS) -> list[Sample]:
    """Collect image and mask pairs for one split.

    Args:
        root: Directory holding the unpacked ``Train``, ``Val`` and ``Test`` trees.
        split: One of ``Train``, ``Val`` or ``Test``.
        domains: Which domains to include; the default takes both.

    The test split ships without labels, so its samples carry no mask.
    """
    root = Path(root)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")

    samples: list[Sample] = []
    for domain in domains:
        image_dir = root / split / domain / "images_png"
        mask_dir = root / split / domain / "masks_png"
        if not image_dir.is_dir():
            continue
        for image in sorted(image_dir.glob("*.png")):
            mask = mask_dir / image.name
            samples.append(Sample(image=image, mask=mask if mask.is_file() else None))

    if not samples:
        raise FileNotFoundError(
            f"no LoveDA imagery under {root / split}; expected {split}/<domain>/images_png/*.png"
        )
    return samples


def prepare(raw_dir: str | Path, out_dir: str | Path, splits: tuple[str, ...] = SPLITS) -> dict:
    """Unpack the downloaded archives and index every split.

    Returns a summary mapping each split to the number of samples found, so the
    caller can report it without walking the tree a second time.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}
    for split in splits:
        archive = raw_dir / f"{split}.zip"
        if not archive.is_file():
            continue
        extract(archive, out_dir)
        try:
            samples = discover(out_dir, split)
        except FileNotFoundError:
            continue
        summary[split] = len(samples)
    if not summary:
        raise FileNotFoundError(
            f"no LoveDA archives in {raw_dir}; download Train.zip and Val.zip from {ZENODO_RECORD}"
        )
    return summary
