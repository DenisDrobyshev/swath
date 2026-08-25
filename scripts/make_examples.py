"""Render a figure of image / ground truth / prediction triples for the README.

Tiles are picked to be different from each other rather than at random: a
figure of six near-identical fields says nothing about a model. The selection
maximises the spread of class composition across the chosen tiles, so a reader
sees the urban case, the agricultural case and the water case rather than three
draws from whatever the corpus happens to be dominated by.

    python scripts/make_examples.py --checkpoint runs/landcover/best.pt \
        --data data/loveda --out docs/examples.png --count 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from swath.checkpoints import load_checkpoint
from swath.data import loveda
from swath.imagery import class_pixel_counts, colorize, read_image, read_mask
from swath.predict import predict_mask
from swath.tasks import Task

PANEL = 256
GAP = 8
LABEL_HEIGHT = 22
LEGEND_HEIGHT = 34
BACKGROUND = (255, 255, 255)
INK = (40, 44, 42)


def composition(mask: np.ndarray, num_classes: int) -> np.ndarray:
    counts = class_pixel_counts(mask, num_classes).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def pick_diverse(profiles: list[np.ndarray], count: int) -> list[int]:
    """Farthest-point selection over class-composition vectors."""
    if count >= len(profiles):
        return list(range(len(profiles)))

    matrix = np.stack(profiles)
    # Start from the tile least like the corpus average.
    centre = matrix.mean(axis=0)
    chosen = [int(np.argmax(((matrix - centre) ** 2).sum(axis=1)))]
    while len(chosen) < count:
        distances = np.min(
            np.stack([((matrix - matrix[index]) ** 2).sum(axis=1) for index in chosen]), axis=0
        )
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return chosen


def to_panel(array: np.ndarray) -> Image.Image:
    image = Image.fromarray(array)
    return image.resize((PANEL, PANEL), Image.LANCZOS if array.ndim == 3 else Image.NEAREST)


def draw_legend(canvas: Image.Image, task: Task, top: int, width: int) -> None:
    draw = ImageDraw.Draw(canvas)
    swatch, spacing = 11, 8
    entries = [(name, task.palette[index]) for index, name in enumerate(task.classes)]
    widths = [swatch + 5 + draw.textlength(name) + spacing * 2 for name, _ in entries]
    x = max(0, (width - sum(widths)) // 2)
    y = top + 10
    for (name, color), entry_width in zip(entries, widths, strict=True):
        draw.rectangle([x, y, x + swatch, y + swatch], fill=tuple(color), outline=(150, 150, 150))
        draw.text((x + swatch + 5, y - 2), name, fill=INK)
        x += entry_width


def build_figure(
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]], task: Task, titles: list[str]
) -> Image.Image:
    columns = 3
    width = columns * PANEL + (columns - 1) * GAP
    height = LABEL_HEIGHT + len(rows) * (PANEL + GAP) - GAP + LEGEND_HEIGHT
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    for column, title in enumerate(titles):
        x = column * (PANEL + GAP)
        draw.text((x + (PANEL - draw.textlength(title)) / 2, 4), title, fill=INK)

    for row_index, panels in enumerate(rows):
        y = LABEL_HEIGHT + row_index * (PANEL + GAP)
        for column, panel in enumerate(panels):
            canvas.paste(to_panel(panel), (column * (PANEL + GAP), y))

    draw_legend(canvas, task, LABEL_HEIGHT + len(rows) * (PANEL + GAP) - GAP, width)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/loveda"))
    parser.add_argument("--split", default="Val")
    parser.add_argument("--out", type=Path, default=Path("docs/examples.png"))
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--pool", type=int, default=160, help="tiles to consider when choosing")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    model, meta = load_checkpoint(args.checkpoint, map_location="cpu")
    task = meta.task
    mapping = loveda.label_map()

    samples = loveda.discover(args.data, args.split)
    step = max(1, len(samples) // args.pool)
    pool = samples[::step][: args.pool]
    print(f"considering {len(pool)} of {len(samples)} tiles")

    profiles = []
    for sample in pool:
        truth = mapping[read_mask(sample.mask)]
        profiles.append(composition(truth, task.num_classes))

    chosen = pick_diverse(profiles, args.count)
    print("chosen tiles:", [pool[index].image.name for index in chosen])

    rows = []
    for index in chosen:
        sample = pool[index]
        image = read_image(sample.image)
        truth = mapping[read_mask(sample.mask)]
        prediction = predict_mask(
            model, image, task, tile=512, overlap=128, device=args.device, tta=args.tta
        )
        rows.append((image, colorize(truth, task.palette), colorize(prediction, task.palette)))

    figure = build_figure(rows, task, ["image", "ground truth", "swath"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.out, optimize=True)
    print(f"{args.out} written ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
