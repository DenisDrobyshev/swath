"""End-to-end smoke test: train, predict, serve — all on synthetic tiles.

The unit suite checks the pieces. This checks that the pieces still fit together
when driven the way a user drives them: a real training run writes a checkpoint,
the CLI loads that checkpoint and segments a raster, and the web service answers
an upload with a mask. It needs no dataset, so it runs in CI on every push.
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from swath.cli import main as cli_main
from swath.data.dataset import Sample, SegmentationDataset
from swath.data.transforms import build_eval_transform, build_train_transform
from swath.tasks import Task
from swath.train import TrainConfig, Trainer

TASK = Task(
    name="smoke",
    title="Smoke task",
    description="Two shapes on a textured background.",
    classes=("background", "square", "disc"),
    palette=((30, 30, 30), (214, 69, 65), (52, 120, 210)),
    mean=(0.5, 0.5, 0.5),
    std=(0.25, 0.25, 0.25),
)


def make_tile(index: int, size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(index)
    mask = np.zeros((size, size), dtype=np.uint8)

    top, left = rng.integers(4, size // 2 - 4, size=2)
    side = int(rng.integers(size // 6, size // 3))
    mask[top : top + side, left : left + side] = 1

    yy, xx = np.mgrid[0:size, 0:size]
    centre = rng.integers(size // 2, size - size // 6, size=2)
    radius = int(rng.integers(size // 10, size // 6))
    mask[(yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 < radius**2] = 2

    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[mask == 0] = (40, 96, 52)
    image[mask == 1] = (208, 72, 64)
    image[mask == 2] = (56, 118, 206)
    noise = rng.integers(-14, 14, image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8), mask


def write_corpus(root: Path, count: int = 16) -> list[Sample]:
    images, masks = root / "images", root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    samples = []
    for index in range(count):
        image, mask = make_tile(index)
        image_path = images / f"tile_{index:02d}.png"
        mask_path = masks / f"tile_{index:02d}.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)
        samples.append(Sample(image=image_path, mask=mask_path))
    return samples


def step(message: str) -> None:
    print(f"\n=== {message}", flush=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        step("building a synthetic corpus")
        samples = write_corpus(root / "corpus")
        print(f"{len(samples)} tiles")

        step("training")
        trainer = Trainer(
            TASK,
            SegmentationDataset(samples[:12], TASK, transform=build_train_transform(64)),
            SegmentationDataset(samples[12:], TASK, transform=build_eval_transform(128)),
            TrainConfig(
                output_dir=root / "run",
                epochs=8,
                batch_size=4,
                crop_size=64,
                val_crop_size=128,
                learning_rate=3e-3,
                base_channels=12,
                depth=3,
                blocks_per_stage=1,
                warmup_epochs=1,
                num_workers=0,
                device="cpu",
                amp=False,
                notes="smoke test",
            ),
        )
        result = trainer.fit()
        best = result["best_mean_iou"]
        print(f"best mIoU {best:.4f}")
        if best < 0.5:
            print("FAIL: a three-colour task should be learnable well past 0.5 mIoU")
            return 1

        checkpoint = root / "run" / "best.pt"

        step("cli: info")
        if cli_main(["info", "--checkpoint", str(checkpoint)]) != 0:
            return 1

        step("cli: predict")
        predictions = root / "predictions"
        code = cli_main(
            [
                "predict",
                "--checkpoint", str(checkpoint),
                "--input", str(root / "corpus" / "images"),
                "--output", str(predictions),
                "--tile", "128",
                "--overlap", "32",
                "--device", "cpu",
                "--overlay",
                "--geojson",
            ]
        )
        if code != 0:
            return 1
        produced = sorted(p.name for p in predictions.iterdir())
        print(f"{len(produced)} files written, e.g. {produced[:3]}")
        if not any(name.endswith("_mask.png") for name in produced):
            print("FAIL: no mask was written")
            return 1

        step("service")
        from fastapi.testclient import TestClient

        from swath.service.app import create_app

        client = TestClient(create_app([checkpoint], device="cpu"))

        health = client.get("/api/health").json()
        print("health:", health)
        if health["models"] != 1:
            print("FAIL: the checkpoint was not loaded")
            return 1

        image, _ = make_tile(99)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        response = client.post(
            "/api/segment",
            files={"file": ("tile.png", buffer.getvalue(), "image/png")},
            data={"tile": "128", "overlap": "32"},
        )
        if response.status_code != 200:
            print(f"FAIL: /api/segment returned {response.status_code}: {response.text[:200]}")
            return 1

        payload = response.json()
        covered = {row["class"]: round(row["share"], 3) for row in payload["classes"]}
        print(f"segmented in {payload['seconds']}s, coverage {covered}")

        download = client.get(payload["downloads"]["mask_png"])
        if download.status_code != 200:
            print("FAIL: the mask download failed")
            return 1

        page = client.get("/")
        if page.status_code != 200 or "swath" not in page.text:
            print("FAIL: the page did not render")
            return 1

    print("\nsmoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
