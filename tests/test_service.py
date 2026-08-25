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
    assert payload["overlay_png"].startswith("data:image/png;base64,")
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
