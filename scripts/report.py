"""Evaluate a checkpoint and emit the numbers as a Markdown table.

Scores the whole validation split at full resolution, then each domain on its
own, and prints the result in the shape a README wants. Splitting by domain is
the point: LoveDA keeps rural and urban apart precisely so that a model which
learned one landscape and collapses on the other cannot hide behind an average.

    python scripts/report.py --checkpoint runs/landcover/best.pt --data data/loveda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swath.checkpoints import load_checkpoint
from swath.data.corpora import get_corpus
from swath.evaluate import EvaluationReport, evaluate


def markdown_table(overall: EvaluationReport, by_domain: dict[str, EvaluationReport]) -> str:
    classes = list(overall.metrics.per_class_iou)
    header = ["class", "IoU", "F1"] + [f"IoU {name.lower()}" for name in by_domain]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    for name in classes:
        row = [
            name,
            f"{overall.metrics.per_class_iou[name]:.3f}",
            f"{overall.metrics.per_class_f1[name]:.3f}",
        ]
        row += [f"{report.metrics.per_class_iou[name]:.3f}" for report in by_domain.values()]
        lines.append("| " + " | ".join(row) + " |")

    mean_row = [
        "**mean**",
        f"**{overall.metrics.mean_iou:.3f}**",
        f"**{overall.metrics.mean_f1:.3f}**",
    ]
    mean_row += [f"**{report.metrics.mean_iou:.3f}**" for report in by_domain.values()]
    lines.append("| " + " | ".join(mean_row) + " |")
    return "\n".join(lines)


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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="write the JSON summary here")
    args = parser.parse_args()

    model, meta = load_checkpoint(args.checkpoint, map_location="cpu")
    corpus = get_corpus(args.dataset)
    shared = {
        "task": meta.task,
        "label_map": corpus.labels(),
        "tile": args.tile,
        "overlap": args.overlap,
        "batch_size": args.batch_size,
        "device": args.device,
        "tta": args.tta,
    }

    def score(domain: str) -> EvaluationReport:
        samples = corpus.samples(args.data, args.split, domain)
        if args.limit:
            samples = samples[: args.limit]
        return evaluate(model, samples, **shared)

    print(f"scoring {args.checkpoint} on {corpus.name} {args.split}")
    overall = score("both")
    by_domain = {domain: score(domain) for domain in corpus.domains}

    print()
    print(overall.summary())
    print()
    print(markdown_table(overall, by_domain))

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": corpus.name,
        "split": args.split,
        "parameters": model.num_parameters(),
        "trained_epochs": meta.epoch,
        "overall": overall.as_dict(),
        "domains": {name: report.as_dict() for name, report in by_domain.items()},
    }
    destination = args.out or args.checkpoint.parent / "report.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwritten to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
