"""Georeferencing: keeping the map coordinates attached to the mask.

A segmentation mask that is only a PNG is a picture of an answer, not the answer
itself — nothing downstream can join it to anything else. When the input carries
a coordinate reference system, this module carries it through to the output, so
the mask comes back as a GeoTIFF that lands in the right place on a map, and as
polygons a GIS can open directly.

rasterio is an optional dependency. Without it the package still trains, predicts
and serves PNG masks; only the georeferenced paths raise.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from swath.tasks import Task

try:  # pragma: no cover - exercised by whichever extras are installed
    import rasterio
    from rasterio import features, warp
    from rasterio.transform import Affine

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False


class MissingGeoSupport(RuntimeError):
    """Raised when a georeferenced operation is attempted without rasterio."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"{operation} needs rasterio; install the geo extra with pip install 'swath[geo]'"
        )


def _require_rasterio(operation: str) -> None:
    if not HAS_RASTERIO:
        raise MissingGeoSupport(operation)


@dataclass(frozen=True)
class GeoReference:
    """Where a raster sits on the Earth."""

    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Pixel width and height in the units of the CRS."""
        a, _, _, _, e, _ = self.transform
        return abs(a), abs(e)

    @property
    def pixel_area(self) -> float:
        width, height = self.pixel_size
        return width * height

    def to_affine(self) -> Any:
        _require_rasterio("building an affine transform")
        return Affine(*self.transform)

    def as_dict(self) -> dict[str, Any]:
        return {
            "crs": self.crs,
            "transform": list(self.transform),
            "width": self.width,
            "height": self.height,
            "pixel_size": list(self.pixel_size),
        }


def read_georeferenced(path: str | Path) -> tuple[np.ndarray, GeoReference | None]:
    """Read a raster along with its georeferencing, when it has any.

    Returns the pixels as ``(H, W, C)`` and a :class:`GeoReference`, or ``None``
    for a plain image such as a JPEG or an ungeoreferenced PNG.
    """
    path = Path(path)
    if not HAS_RASTERIO:
        from swath.imagery import read_image

        return read_image(path), None

    try:
        with warnings.catch_warnings():
            # A plain PNG or JPEG has no geotransform; that is expected here and
            # is answered with reference=None rather than a warning.
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            dataset = rasterio.open(path)
        with dataset:
            array = dataset.read().transpose(1, 2, 0)
            crs = dataset.crs
            transform = dataset.transform
            georeferenced = crs is not None and not transform.is_identity
            reference = (
                GeoReference(
                    crs=crs.to_string(),
                    transform=tuple(transform)[:6],
                    width=dataset.width,
                    height=dataset.height,
                )
                if georeferenced
                else None
            )
    except Exception:
        from swath.imagery import read_image

        return read_image(path), None

    from swath.imagery import _percentile_stretch

    if array.dtype != np.uint8:
        array = _percentile_stretch(array)
    return np.ascontiguousarray(array), reference


def write_mask_geotiff(
    path: str | Path,
    mask: np.ndarray,
    reference: GeoReference,
    task: Task | None = None,
    compress: str = "deflate",
) -> Path:
    """Write a label map as a single-band GeoTIFF.

    The task palette is written as a TIFF colour table, so the file opens in QGIS
    already coloured instead of as an unreadable ramp of small integers.
    """
    _require_rasterio("writing a GeoTIFF")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": reference.crs,
        "transform": reference.to_affine(),
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(mask.astype(np.uint8), 1)
        if task is not None:
            dataset.write_colormap(
                1, {index: tuple(color) for index, color in enumerate(task.palette)}
            )
    return path


def sieve_mask(mask: np.ndarray, min_pixels: int, connectivity: int = 4) -> np.ndarray:
    """Remove connected regions smaller than ``min_pixels``.

    Any per-pixel classifier leaves speckle: single pixels and small clumps that
    disagree with everything around them. Sieving reassigns each such region to
    its largest neighbour, which is the standard cleanup in remote sensing and
    is what makes a mask usable as polygons rather than as a picture.

    It is off by default and should stay off when small objects are the point —
    a sieve large enough to tidy a land-cover map will also erase a single
    parked car or an isolated shed.
    """
    _require_rasterio("sieving a mask")
    if min_pixels <= 1:
        return mask
    return features.sieve(mask.astype(np.uint8), size=int(min_pixels), connectivity=connectivity)


def mask_to_geojson(
    mask: np.ndarray,
    reference: GeoReference | None,
    task: Task,
    *,
    skip_classes: tuple[int, ...] = (0,),
    min_pixels: int = 32,
    to_wgs84: bool = True,
) -> dict[str, Any]:
    """Vectorise a label map into a GeoJSON ``FeatureCollection``.

    Args:
        mask: Label map.
        reference: Georeferencing; without it polygons come out in pixel coordinates.
        task: Supplies class names and colours for the feature properties.
        skip_classes: Class indices not worth vectorising — background by default.
        min_pixels: Drop polygons smaller than this, which removes the speckle
            that any per-pixel classifier leaves behind.
        to_wgs84: Reproject to EPSG:4326, the coordinate system GeoJSON assumes.
    """
    _require_rasterio("vectorising a mask")

    transform = reference.to_affine() if reference else Affine.identity()
    valid = np.isin(mask, list(skip_classes), invert=True)
    shapes = features.shapes(mask.astype(np.uint8), mask=valid, transform=transform, connectivity=4)

    pixel_area = reference.pixel_area if reference else 1.0
    minimum_area = min_pixels * pixel_area

    collection: list[dict[str, Any]] = []
    for geometry, value in shapes:
        index = int(value)
        if index >= task.num_classes:
            continue
        area = _polygon_area(geometry)
        if area < minimum_area:
            continue
        output_geometry = geometry
        if to_wgs84 and reference is not None:
            output_geometry = warp.transform_geom(reference.crs, "EPSG:4326", geometry)
        collection.append(
            {
                "type": "Feature",
                "geometry": output_geometry,
                "properties": {
                    "class_index": index,
                    "class": task.classes[index],
                    "color": "#{:02x}{:02x}{:02x}".format(*task.palette[index]),
                    "area": round(area, 3),
                    "area_unit": "map units squared" if reference else "pixels",
                },
            }
        )

    collection.sort(key=lambda feature: feature["properties"]["area"], reverse=True)
    return {
        "type": "FeatureCollection",
        "crs_hint": _crs_hint(reference, to_wgs84),
        "features": collection,
    }


def _polygon_area(geometry: dict[str, Any]) -> float:
    """Area of a GeoJSON polygon by the shoelace formula, holes subtracted."""
    if geometry.get("type") != "Polygon":
        return 0.0
    rings = geometry.get("coordinates", [])
    if not rings:
        return 0.0
    area = abs(_ring_area(rings[0]))
    for hole in rings[1:]:
        area -= abs(_ring_area(hole))
    return max(area, 0.0)


def _ring_area(ring: list[tuple[float, float]]) -> float:
    if len(ring) < 3:
        return 0.0
    coordinates = np.asarray(ring, dtype=np.float64)
    x, y = coordinates[:, 0], coordinates[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def class_areas(
    mask: np.ndarray, task: Task, reference: GeoReference | None = None
) -> list[dict[str, Any]]:
    """Per-class coverage of a mask, in pixels and in map units where known."""
    from swath.imagery import class_pixel_counts

    counts = class_pixel_counts(mask, task.num_classes)
    total = int(counts.sum())
    pixel_area = reference.pixel_area if reference else None

    rows: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        row: dict[str, Any] = {
            "class_index": index,
            "class": task.classes[index],
            "color": "#{:02x}{:02x}{:02x}".format(*task.palette[index]),
            "pixels": int(count),
            "share": round(float(count) / total, 6) if total else 0.0,
        }
        if pixel_area:
            row["area_m2"] = round(float(count) * pixel_area, 2)
        rows.append(row)
    return rows


def _crs_hint(reference: GeoReference | None, to_wgs84: bool) -> str | None:
    """Which coordinate system the vectorised output is expressed in."""
    if reference is None:
        return None
    return "EPSG:4326" if to_wgs84 else reference.crs
