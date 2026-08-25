"""Evaluate a checkpoint and emit the numbers as a Markdown table.

Scores the whole validation split at full resolution and reports each domain
separately. Splitting by domain is the point: LoveDA keeps rural and urban apart
precisely so that a model which learned one landscape and collapses on the other
cannot hide behind an average.

Each tile is segmented once. A confusion matrix is accumulated per domain and the
overall figures come from their sum, rather than from a second and third pass
over the same imagery.

    python scripts/report.py --checkpoint runs/landcover/best.pt --data data/loveda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from swath.checkpoints import load_checkpoint
from swath.data.corpora import get_corpus
from swath.imagery import read_image, read_mask
from swath.metrics import ConfusionMatrix, MetricResult
from swath.predict import predict_mask, select_device
from swath.tasks import Task


def score_domains(
    model,
    corpus,
    task: Task,
    data: Path,
    split: str,
    *,
    tile: int,
    overlap: int,
    batch_size: int,
    device: str,
    tta: bool,
    sieve: int,
    limit: int,
) -> tuple[MetricResult, dict[str, MetricResult], int]:
    """Segment every tile once, accumulating a confusion matrix per domain."""
    resolved = select_device(device)
    model = model.to(resolved).eval()
    label_map = corpus.labels()

    domains = corpus.domains or ("all",)
    matrices = {name: ConfusionMatrix(task.num_classes, device=resolved) for name in domains}
    overall = ConfusionMatrix(task.num_classes, device=resolved)

    seen = 0
    started = time.time()
    for domain in domains:
        samples = corpus.samples(data, split, domain if corpus.domains else "both")
        samples = [sample for sample in samples if sample.mask is not None]
        if limit:
            samples = samples[:limit]

        for index, sample in enumerate(samples):
            image = read_image(sample.image)
            truth = read_mask(sample.mask)
            if label_map is not None:
                truth = label_map[truth]

            with torch.inference_mode():
                prediction = predict_mask(
                    model,
                    image,
                    task,
                    tile=tile,
                    overlap=overlap,
                    batch_size=batch_size,
                    device=resolved,
                    tta=tta,
                    sieve=sieve,
                )

            truth_tensor = torch.from_numpy(truth.astype(np.int64))
            prediction_tensor = torch.from_numpy(prediction.astype(np.int64))
            matrices[domain].update(truth_tensor, prediction_tensor)
            overall.update(truth_tensor, prediction_tensor)
            seen += 1

            if index % 50 == 0 or index == len(samples) - 1:
                rate = seen / max(time.time() - started, 1e-6)
                print(
                    f"  {domain}: {index + 1}/{len(samples)}  {rate:.1f} tiles/s",
                    flush=True,
                )

    return (
        overall.compute(task.classes),
        {name: matrix.compute(task.classes) for name, matrix in matrices.items()},
        seen,
    )


def markdown_table(overall: MetricResult, by_domain: dict[str, MetricResult]) -> str:
    header = ["class", "IoU", "F1"] + [f"IoU {name.lower()}" for name in by_domain]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    for name in overall.per_class_iou:
        row = [name, f"{overall.per_class_iou[name]:.3f}", f"{overall.per_class_f1[name]:.3f}"]
        row += [f"{result.per_class_iou[name]:.3f}" for result in by_domain.values()]
        lines.append("| " + " | ".join(row) + " |")

    mean_row = ["**mean**", f"**{overall.mean_iou:.3f}**", f"**{overall.mean_f1:.3f}**"]
    mean_row += [f"**{result.mean_iou:.3f}**" for result in by_domain.values()]
    lines.append("| " + " | ".join(mean_row) + " |")
    return "\n".join(lines)


def as_dict(result: MetricResult) -> dict:
    return {
        "overall_accuracy": round(result.overall_accuracy, 5),
        "mean_iou": round(result.mean_iou, 5),
        "mean_f1": round(result.mean_f1, 5),
        "per_class_iou": {k: round(v, 5) for k, v in result.per_class_iou.items()},
        "per_class_f1": {k: round(v, 5) for k, v in result.per_class_f1.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="loveda")
    parser.add_argument("--data", type=Path, default=Path("data/loveda"))
    parser.add_argument("--split", default="Val")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--sieve", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="write the JSON summary here")
    args = parser.parse_args()

    model, meta = load_checkpoint(args.checkpoint, map_location="cpu")
    corpus = get_corpus(args.dataset)

    print(f"scoring {args.checkpoint} on {corpus.name} {args.split}", flush=True)
    started = time.time()
    overall, by_domain, seen = score_domains(
        model,
        corpus,
        meta.task,
        args.data,
        args.split,
        tile=args.tile,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=args.device,
        tta=args.tta,
        sieve=args.sieve,
        limit=args.limit,
    )
    elapsed = time.time() - started

    print(f"\n{seen} tiles in {elapsed / 60:.1f} min")
    print(overall.summary())
    print()
    print(overall.table())
    print()
    print(markdown_table(overall, by_domain))

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": corpus.name,
        "split": args.split,
        "tiles": seen,
        "tile": args.tile,
        "overlap": args.overlap,
        "tta": args.tta,
        "sieve": args.sieve,
        "parameters": model.num_parameters(),
        "trained_epochs": meta.epoch,
        "overall": as_dict(overall),
        "domains": {name: as_dict(result) for name, result in by_domain.items()},
    }
    destination = args.out or args.checkpoint.parent / "report.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwritten to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
