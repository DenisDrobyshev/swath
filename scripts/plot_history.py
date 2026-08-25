"""Plot a training run from its ``history.csv``.

Two panels: the loss curves, and mean IoU with the per-class spread behind it.
The spread is the part worth showing — on land cover a mean that climbs while one
class stays flat at the bottom is the normal outcome, and a single averaged line
hides exactly that.

    python scripts/plot_history.py --run runs/landcover --out docs/training.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

INK = "#1b1f1c"
MUTED = "#8b938c"
GRID = "#e4e7e3"
ACCENT = "#146b52"
WARM = "#b4622f"


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def series(rows: list[dict[str, str]], column: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(column, "")
        if value not in ("", None):
            xs.append(float(row["epoch"]))
            ys.append(float(value))
    return xs, ys


def style(axis: plt.Axes) -> None:
    from matplotlib.ticker import MaxNLocator

    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_facecolor("white")
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/landcover"))
    parser.add_argument("--out", type=Path, default=Path("docs/training.png"))
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    rows = read_history(args.run / "history.csv")
    if not rows:
        print("history.csv is empty")
        return 1

    class_columns = [key for key in rows[0] if key.startswith("iou_")]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.1), dpi=160)
    figure.patch.set_facecolor("white")

    epochs, train_loss = series(rows, "train_loss")
    left.plot(epochs, train_loss, color=ACCENT, linewidth=1.8, label="train")
    val_epochs, val_loss = series(rows, "val_loss")
    if val_loss:
        left.plot(val_epochs, val_loss, color=WARM, linewidth=1.8, label="validation")
    left.set_xlabel("epoch")
    left.set_ylabel("loss")
    left.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    style(left)

    for column in class_columns:
        xs, ys = series(rows, column)
        if ys:
            right.plot(xs, ys, color=MUTED, linewidth=0.9, alpha=0.55)
    miou_epochs, miou = series(rows, "val_miou")
    right.plot(miou_epochs, miou, color=ACCENT, linewidth=2.2, label="mean IoU")
    if miou:
        best = max(range(len(miou)), key=lambda i: miou[i])
        right.scatter([miou_epochs[best]], [miou[best]], color=ACCENT, s=26, zorder=3)
        right.annotate(
            f"{miou[best]:.3f}",
            (miou_epochs[best], miou[best]),
            textcoords="offset points",
            xytext=(6, -3),
            color=INK,
            fontsize=9,
        )
    right.set_xlabel("epoch")
    right.set_ylabel("IoU")
    right.set_ylim(0, max(0.6, (max(miou) if miou else 0.6) * 1.25))
    right.legend(
        frameon=False,
        fontsize=9,
        labelcolor=MUTED,
        handles=[
            plt.Line2D([], [], color=ACCENT, linewidth=2.2, label="mean"),
            plt.Line2D([], [], color=MUTED, linewidth=0.9, label="per class"),
        ],
    )
    style(right)

    if args.title:
        figure.suptitle(args.title, color=INK, fontsize=11, y=0.99)

    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, facecolor="white", bbox_inches="tight")
    print(f"{args.out} written ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
