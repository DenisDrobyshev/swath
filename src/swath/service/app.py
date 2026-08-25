"""FastAPI application serving one or more segmentation checkpoints.

The service is intentionally stateless apart from a small results cache: an
upload is segmented in the request, the mask is returned inline as a PNG, and
the heavier artefacts — the GeoTIFF and the vectorised polygons — are kept
briefly under a result id so the page can offer them as ordinary download links
instead of pushing megabytes of base64 into the JSON response.
"""

from __future__ import annotations

import base64
import io
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from swath import __version__
from swath.checkpoints import load_checkpoint
from swath.geo import HAS_RASTERIO, GeoReference, class_areas, mask_to_geojson
from swath.imagery import colorize, overlay
from swath.models import UNet
from swath.predict import predict_mask, select_device
from swath.tasks import Task

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 64_000_000
RESULT_CACHE_SIZE = 24


@dataclass
class LoadedModel:
    """A checkpoint held in memory, ready to serve."""

    identifier: str
    model: UNet
    task: Task
    metrics: dict[str, float]
    source: str

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "task": self.task.name,
            "title": self.task.title,
            "description": self.task.description,
            "bands": self.task.in_channels,
            "parameters": self.model.num_parameters(),
            "metrics": {key: round(value, 4) for key, value in self.metrics.items()},
            "source": self.source,
            "classes": [
                {
                    "index": index,
                    "name": name,
                    "color": "#{:02x}{:02x}{:02x}".format(*self.task.palette[index]),
                }
                for index, name in enumerate(self.task.classes)
            ],
        }


class ResultCache:
    """A tiny bounded store for artefacts the page may ask to download."""

    def __init__(self, capacity: int = RESULT_CACHE_SIZE) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def put(self, payload: dict[str, Any]) -> str:
        identifier = uuid.uuid4().hex[:16]
        self._items[identifier] = payload | {"created": time.time()}
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return identifier

    def get(self, identifier: str) -> dict[str, Any] | None:
        item = self._items.get(identifier)
        if item is not None:
            self._items.move_to_end(identifier)
        return item


def _encode_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _read_upload(data: bytes, filename: str) -> tuple[np.ndarray, GeoReference | None]:
    """Decode an upload, preferring rasterio so georeferencing survives."""
    suffix = Path(filename).suffix.lower()
    if HAS_RASTERIO and suffix in {".tif", ".tiff"}:
        from rasterio.io import MemoryFile

        with MemoryFile(data) as memory, memory.open() as dataset:
            array = dataset.read().transpose(1, 2, 0)
            reference = None
            if dataset.crs is not None and not dataset.transform.is_identity:
                reference = GeoReference(
                    crs=dataset.crs.to_string(),
                    transform=tuple(dataset.transform)[:6],
                    width=dataset.width,
                    height=dataset.height,
                )
        if array.dtype != np.uint8:
            from swath.imagery import _percentile_stretch

            array = _percentile_stretch(array)
        return np.ascontiguousarray(array), reference

    with Image.open(io.BytesIO(data)) as handle:
        handle = handle.convert("RGB") if handle.mode not in {"RGB", "L"} else handle
        array = np.asarray(handle)
    if array.ndim == 2:
        array = array[:, :, None]
    return np.ascontiguousarray(array), None


def _fit_bands(image: np.ndarray, expected: int) -> np.ndarray:
    channels = image.shape[2]
    if channels == expected:
        return image
    if channels == 1 and expected > 1:
        return np.repeat(image, expected, axis=2)
    if channels > expected:
        return image[:, :, :expected]
    raise HTTPException(
        status_code=422,
        detail=f"image has {channels} bands but the model expects {expected}",
    )


def discover_checkpoints(paths: list[Path] | None) -> list[Path]:
    """Expand the checkpoints given on the command line.

    A directory contributes every ``.pt`` file inside it, which makes
    ``swath serve --checkpoint runs`` do the obvious thing.
    """
    found: list[Path] = []
    for path in paths or []:
        path = Path(path)
        if path.is_dir():
            found.extend(sorted(path.rglob("*.pt")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"checkpoint {path} does not exist")
    return found


def create_app(
    checkpoints: list[Path] | None = None,
    device: str = "auto",
    max_pixels: int = MAX_PIXELS,
) -> FastAPI:
    """Build the application around the given checkpoints."""
    resolved = discover_checkpoints(checkpoints)
    torch_device = select_device(device)

    models: dict[str, LoadedModel] = {}
    for path in resolved:
        model, meta = load_checkpoint(path, map_location=torch_device)
        model.to(torch_device).eval()
        identifier = path.parent.name if path.stem in {"best", "last"} else path.stem
        base = identifier
        counter = 2
        while identifier in models:
            identifier = f"{base}-{counter}"
            counter += 1
        models[identifier] = LoadedModel(
            identifier=identifier,
            model=model,
            task=meta.task,
            metrics=meta.metrics,
            source=path.name,
        )

    cache = ResultCache()
    app = FastAPI(
        title="swath",
        version=__version__,
        description="Semantic segmentation of aerial and satellite imagery.",
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "device": str(torch_device),
            "models": len(models),
            "geo_support": HAS_RASTERIO,
        }

    @app.get("/api/models")
    def list_models() -> dict[str, Any]:
        return {"models": [model.describe() for model in models.values()]}

    @app.post("/api/segment")
    async def segment(
        file: UploadFile = File(...),
        model_id: str = Form(""),
        tile: int = Form(512),
        overlap: int = Form(128),
        alpha: float = Form(0.5),
        tta: bool = Form(False),
    ) -> JSONResponse:
        if not models:
            raise HTTPException(status_code=503, detail="no checkpoints are loaded")

        entry = models.get(model_id) if model_id else next(iter(models.values()))
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown model {model_id!r}")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=422, detail="the uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            )

        try:
            image, reference = _read_upload(data, file.filename or "upload")
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=422, detail=f"cannot read the image: {error}"
            ) from error

        height, width = image.shape[:2]
        if height * width > max_pixels:
            limit = max_pixels // 1_000_000
            raise HTTPException(
                status_code=413,
                detail=f"image is {width}x{height}; the limit is {limit} megapixels",
            )

        image = _fit_bands(image, entry.task.in_channels)

        started = time.time()
        with torch.inference_mode():
            mask = predict_mask(
                entry.model,
                image,
                entry.task,
                tile=max(64, int(tile)),
                overlap=max(0, min(int(overlap), int(tile) - 32)),
                batch_size=4,
                device=torch_device,
                tta=bool(tta),
            )
        elapsed = time.time() - started

        areas = class_areas(mask, entry.task, reference)
        mask_png = _encode_png(colorize(mask, entry.task.palette))
        overlay_png = _encode_png(overlay(image, mask, entry.task.palette, alpha=alpha))

        payload: dict[str, Any] = {
            "mask": mask,
            "task": entry.task,
            "reference": reference,
            "mask_png": mask_png,
            "filename": Path(file.filename or "upload").stem,
        }
        result_id = cache.put(payload)

        return JSONResponse(
            {
                "result_id": result_id,
                "model": entry.identifier,
                "task": entry.task.name,
                "width": int(width),
                "height": int(height),
                "seconds": round(elapsed, 3),
                "georeferenced": reference is not None,
                "geo": reference.as_dict() if reference else None,
                "classes": areas,
                "mask_png": "data:image/png;base64," + base64.b64encode(mask_png).decode(),
                "overlay_png": "data:image/png;base64," + base64.b64encode(overlay_png).decode(),
                "downloads": _downloads(result_id, reference is not None),
            }
        )

    def _downloads(result_id: str, georeferenced: bool) -> dict[str, str]:
        links = {"mask_png": f"/api/result/{result_id}/mask.png"}
        if HAS_RASTERIO:
            links["geojson"] = f"/api/result/{result_id}/mask.geojson"
            if georeferenced:
                links["geotiff"] = f"/api/result/{result_id}/mask.tif"
        return links

    @app.get("/api/result/{result_id}/mask.png")
    def download_mask(result_id: str) -> Response:
        item = _require_result(result_id)
        return Response(
            content=item["mask_png"],
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{item["filename"]}_mask.png"'
            },
        )

    @app.get("/api/result/{result_id}/mask.geojson")
    def download_geojson(result_id: str) -> JSONResponse:
        item = _require_result(result_id)
        if not HAS_RASTERIO:
            raise HTTPException(status_code=501, detail="vectorising needs the geo extra")
        payload = mask_to_geojson(item["mask"], item["reference"], item["task"])
        return JSONResponse(
            payload,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{item["filename"]}.geojson"'
                )
            },
        )

    @app.get("/api/result/{result_id}/mask.tif")
    def download_geotiff(result_id: str) -> FileResponse:
        item = _require_result(result_id)
        if item["reference"] is None:
            raise HTTPException(status_code=404, detail="the input was not georeferenced")
        from tempfile import NamedTemporaryFile

        from swath.geo import write_mask_geotiff

        # The file has to outlive this handler: FileResponse streams it after
        # the function returns, so it cannot be opened as a context manager.
        handle = NamedTemporaryFile(suffix=".tif", delete=False)  # noqa: SIM115
        handle.close()
        write_mask_geotiff(handle.name, item["mask"], item["reference"], item["task"])
        return FileResponse(
            handle.name,
            media_type="image/tiff",
            filename=f"{item['filename']}_mask.tif",
        )

    def _require_result(result_id: str) -> dict[str, Any]:
        item = cache.get(result_id)
        if item is None:
            raise HTTPException(
                status_code=404, detail="this result has expired; segment the image again"
            )
        return item

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app
