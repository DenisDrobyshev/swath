from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swath.geo import HAS_RASTERIO, GeoReference, class_areas, mask_to_geojson

pytestmark = pytest.mark.skipif(not HAS_RASTERIO, reason="needs the geo extra")


@pytest.fixture
def reference() -> GeoReference:
    # A metre-per-pixel grid in UTM zone 37N, origin at a round easting/northing.
    return GeoReference(
        crs="EPSG:32637",
        transform=(1.0, 0.0, 400000.0, 0.0, -1.0, 6200000.0),
        width=64,
        height=64,
    )


@pytest.fixture
def mask() -> np.ndarray:
    array = np.zeros((64, 64), dtype=np.uint8)
    array[8:24, 8:24] = 1  # a 16x16 square of class 1
    array[40:48, 40:56] = 2
    return array


def test_pixel_area_comes_from_the_transform(reference: GeoReference):
    assert reference.pixel_size == (1.0, 1.0)
    assert reference.pixel_area == 1.0


def test_geotiff_keeps_crs_and_palette(tmp_path: Path, mask, reference, task):
    import rasterio

    from swath.geo import write_mask_geotiff

    path = write_mask_geotiff(tmp_path / "mask.tif", mask, reference, task)
    with rasterio.open(path) as dataset:
        assert dataset.crs.to_epsg() == 32637
        assert dataset.count == 1
        assert np.array_equal(dataset.read(1), mask)
        assert dataset.colormap(1)[1][:3] == task.palette[1]


def test_vectorising_finds_the_squares(mask, reference, task):
    payload = mask_to_geojson(mask, reference, task, min_pixels=4, to_wgs84=False)
    assert payload["type"] == "FeatureCollection"
    classes = {feature["properties"]["class"] for feature in payload["features"]}
    assert classes == {"stripe", "blob"}

    stripe = next(f for f in payload["features"] if f["properties"]["class"] == "stripe")
    assert stripe["properties"]["area"] == pytest.approx(16 * 16, rel=1e-6)


def test_background_is_skipped_by_default(mask, reference, task):
    payload = mask_to_geojson(mask, reference, task, to_wgs84=False)
    assert all(f["properties"]["class_index"] != 0 for f in payload["features"])


def test_small_polygons_are_dropped(reference, task):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[0, 0] = 1  # a single speckled pixel
    payload = mask_to_geojson(mask, reference, task, min_pixels=32, to_wgs84=False)
    assert payload["features"] == []


def test_reprojection_to_wgs84_produces_degrees(mask, reference, task):
    payload = mask_to_geojson(mask, reference, task, min_pixels=4)
    assert payload["crs_hint"] == "EPSG:4326"
    longitude, latitude = payload["features"][0]["geometry"]["coordinates"][0][0]
    assert -180 <= longitude <= 180
    assert -90 <= latitude <= 90


def test_class_areas_report_square_metres(mask, reference, task):
    rows = class_areas(mask, task, reference)
    stripe = next(row for row in rows if row["class"] == "stripe")
    assert stripe["pixels"] == 16 * 16
    assert stripe["area_m2"] == pytest.approx(256.0)
    assert sum(row["share"] for row in rows) == pytest.approx(1.0)


def test_class_areas_without_georeferencing_stay_in_pixels(mask, task):
    rows = class_areas(mask, task, None)
    assert all("area_m2" not in row for row in rows)
