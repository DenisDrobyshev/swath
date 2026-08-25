from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from swath.service.app import create_app, discover_checkpoints


@pytest.fixture
def client(checkpoint: Path) -> TestClient:
    return TestClient(create_app([checkpoint], device="cpu"))


def _png_upload(size: int = 96) -> bytes:
    rng = np.random.default_rng(3)
    array = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_a_loaded_model(client: TestClient):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["models"] == 1


def test_models_endpoint_describes_the_classes(client: TestClient):
    models = client.get("/api/models").json()["models"]
    assert len(models) == 1
    assert [entry["name"] for entry in models[0]["classes"]] == [
        "background",
        "stripe",
        "blob",
    ]
    assert models[0]["classes"][1]["color"] == "#dc2828"


def test_segment_returns_a_mask_and_coverage(client: TestClient):
    response = client.post(
        "/api/segment",
        files={"file": ("tile.png", _png_upload(), "image/png")},
        data={"tile": "64", "overlap": "16"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["width"] == 96 and payload["height"] == 96
    assert payload["mask_png"].startswith("data:image/png;base64,")
    assert payload["image_png"].startswith("data:image/png;base64,")
    assert abs(sum(row["share"] for row in payload["classes"]) - 1.0) < 1e-4
    assert payload["georeferenced"] is False


def test_mask_download_round_trips(client: TestClient):
    payload = client.post(
        "/api/segment",
        files={"file": ("tile.png", _png_upload(), "image/png")},
        data={"tile": "64", "overlap": "16"},
    ).json()

    response = client.get(payload["downloads"]["mask_png"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as handle:
        assert handle.size == (96, 96)


def test_empty_upload_is_rejected(client: TestClient):
    response = client.post("/api/segment", files={"file": ("tile.png", b"", "image/png")})
    assert response.status_code == 422


def test_unreadable_upload_is_rejected(client: TestClient):
    response = client.post(
        "/api/segment", files={"file": ("tile.png", b"not an image", "image/png")}
    )
    assert response.status_code == 422


def test_unknown_model_is_a_404(client: TestClient):
    response = client.post(
        "/api/segment",
        files={"file": ("tile.png", _png_upload(), "image/png")},
        data={"model_id": "nope"},
    )
    assert response.status_code == 404


def test_oversized_image_is_refused(checkpoint: Path):
    client = TestClient(create_app([checkpoint], device="cpu", max_pixels=1000))
    response = client.post(
        "/api/segment", files={"file": ("tile.png", _png_upload(96), "image/png")}
    )
    assert response.status_code == 413


def test_expired_result_is_reported(client: TestClient):
    assert client.get("/api/result/deadbeef/mask.png").status_code == 404


def test_index_page_is_served(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "swath" in response.text


def test_service_without_checkpoints_says_so():
    client = TestClient(create_app([], device="cpu"))
    assert client.get("/api/health").json()["models"] == 0
    response = client.post("/api/segment", files={"file": ("t.png", _png_upload(), "image/png")})
    assert response.status_code == 503


def test_discover_expands_a_directory(checkpoint: Path):
    found = discover_checkpoints([checkpoint.parent])
    assert checkpoint in found


def test_discover_rejects_a_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_checkpoints([tmp_path / "nope.pt"])


def _decode_data_url(url: str) -> Image.Image:
    import base64

    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_previews_are_downscaled_but_downloads_are_not(checkpoint: Path):
    """A large mask must not travel to the browser at full resolution."""
    client = TestClient(create_app([checkpoint], device="cpu"))
    payload = client.post(
        "/api/segment",
        files={"file": ("tile.png", _png_upload(320), "image/png")},
        data={"tile": "64", "overlap": "16"},
    ).json()

    # PREVIEW_MAX_SIDE is 2048 in production; patching it would not exercise the
    # real path, so the check is that previews never exceed it and that the
    # download keeps the original size.
    preview = _decode_data_url(payload["mask_png"])
    assert max(preview.size) <= payload["preview_max_side"]

    download = client.get(payload["downloads"]["mask_png"])
    with Image.open(io.BytesIO(download.content)) as full:
        assert full.size == (320, 320)


def test_preview_downscaling_kicks_in(checkpoint: Path):
    client = TestClient(create_app([checkpoint], device="cpu", preview_max_side=64))
    payload = client.post(
        "/api/segment",
        files={"file": ("tile.png", _png_upload(160), "image/png")},
        data={"tile": "64", "overlap": "16"},
    ).json()

    assert max(_decode_data_url(payload["mask_png"]).size) == 64
    assert max(_decode_data_url(payload["image_png"]).size) == 64
    assert payload["width"] == 160, "the reported size is the raster, not the preview"


def test_mask_preview_keeps_palette_colours_exact(checkpoint: Path):
    """Nearest-neighbour resampling: a downscaled mask must invent no new colours."""
    client = TestClient(create_app([checkpoint], device="cpu"))
    payload = client.post(
        "/api/segment",
        files={"file": ("tile.png", _png_upload(256), "image/png")},
        data={"tile": "64", "overlap": "16"},
    ).json()

    preview = np.asarray(_decode_data_url(payload["mask_png"]).convert("RGB"))
    colours = {tuple(colour) for colour in preview.reshape(-1, 3)}
    models = client.get("/api/models").json()["models"][0]
    allowed = {
        tuple(int(entry["color"][i : i + 2], 16) for i in (1, 3, 5)) for entry in models["classes"]
    }
    assert colours <= allowed


def test_index_stamps_a_version_onto_the_assets(client: TestClient):
    from swath.service.app import asset_version

    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert "__SWATH_ASSETS__" not in response.text
    assert f"/app.js?v={asset_version()}" in response.text
    assert f"/style.css?v={asset_version()}" in response.text


def test_asset_version_follows_the_files(tmp_path: Path):
    import os

    from swath.service.app import asset_version

    (tmp_path / "app.js").write_text("one", encoding="utf-8")
    first = asset_version(tmp_path)

    (tmp_path / "app.js").write_text("two but longer", encoding="utf-8")
    os.utime(tmp_path / "app.js", (1, 1))
    assert asset_version(tmp_path) != first


def test_static_assets_are_still_served(client: TestClient):
    for path in ("/app.js", "/style.css"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.content


def test_segmentation_does_not_block_the_event_loop(checkpoint: Path):
    """A slow segmentation must not stop the service answering anything else.

    Timing has to be measured against a clock started *before* the segmentation,
    not after it is seen to begin: a blocking handler freezes the event loop, so
    any measurement taken once the loop is running again reports a fast health
    check no matter how long the freeze lasted.
    """
    import asyncio
    import threading
    import time as time_module

    import httpx

    from swath.service import app as service_app

    hold = 1.0
    started = threading.Event()
    real_predict = service_app.predict_mask

    def slow_predict(*args, **kwargs):
        started.set()
        time_module.sleep(hold)
        return real_predict(*args, **kwargs)

    service_app.predict_mask = slow_predict
    try:
        application = service_app.create_app([checkpoint], device="cpu")

        async def exercise() -> tuple[float, httpx.Response, httpx.Response]:
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                begin = time_module.perf_counter()
                task = asyncio.create_task(
                    client.post(
                        "/api/segment",
                        files={"file": ("tile.png", _png_upload(), "image/png")},
                        data={"tile": "64", "overlap": "16"},
                    )
                )
                await asyncio.sleep(0.1)  # let the upload reach the handler
                health = await client.get("/api/health")
                answered_at = time_module.perf_counter() - begin
                return answered_at, health, await task

        answered_at, health, response = asyncio.run(exercise())

        assert started.is_set(), "the segmentation never reached the model"
        assert health.status_code == 200
        assert response.status_code == 200
        assert answered_at < hold / 2, (
            f"the health check was answered {answered_at:.2f}s in, behind a "
            f"{hold:.1f}s segmentation"
        )
    finally:
        service_app.predict_mask = real_predict
