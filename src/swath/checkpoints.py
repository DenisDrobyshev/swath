"""Saving and restoring trained models.

A checkpoint here is self-describing: alongside the weights it stores the
architecture arguments and the full task definition, class names and palette
included. Loading one therefore needs nothing but the file — no matching config,
no assumption that the registry still holds the same task under the same name.
That is what lets the web service accept a checkpoint it has never seen and
still label and colour its output correctly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from swath import __version__
from swath.models import UNet, build_model
from swath.tasks import Task

FORMAT_VERSION = 1


@dataclass
class CheckpointMeta:
    """Everything a checkpoint carries besides the weights."""

    task: Task
    model: dict[str, Any]
    epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    swath_version: str = __version__
    format_version: int = FORMAT_VERSION
    notes: str = ""


def model_config(model: UNet) -> dict[str, Any]:
    """Recover the arguments needed to rebuild ``model``."""
    first_decoder = model.decoder[0]
    return {
        "in_channels": model.in_channels,
        "num_classes": model.num_classes,
        "base_channels": model.stem[0].out_channels,
        "depth": model.depth,
        "blocks_per_stage": len(first_decoder.blocks),
        "norm": "group" if isinstance(model.stem[1], torch.nn.GroupNorm) else "batch",
        "dropout": float(getattr(model.dropout, "p", 0.0)),
    }


def save_checkpoint(
    path: str | Path,
    model: UNet,
    task: Task,
    *,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    history: list[dict[str, float]] | None = None,
    notes: str = "",
) -> Path:
    """Write weights, architecture and task definition to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "swath_version": __version__,
        "task": asdict(task),
        "model": model_config(model),
        "epoch": epoch,
        "metrics": metrics or {},
        "history": history or [],
        "notes": notes,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[UNet, CheckpointMeta]:
    """Rebuild the model and its metadata from a checkpoint file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint {path} does not exist")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    format_version = int(payload.get("format_version", 0))
    if format_version > FORMAT_VERSION:
        raise ValueError(
            f"{path.name} was written by a newer swath (checkpoint format "
            f"{format_version}, this build reads {FORMAT_VERSION})"
        )

    task_payload = dict(payload["task"])
    task_payload["classes"] = tuple(task_payload["classes"])
    task_payload["palette"] = tuple(tuple(color) for color in task_payload["palette"])
    task_payload["mean"] = tuple(task_payload["mean"])
    task_payload["std"] = tuple(task_payload["std"])
    task = Task(**task_payload)

    config = dict(payload["model"])
    model = build_model(**config)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    meta = CheckpointMeta(
        task=task,
        model=config,
        epoch=int(payload.get("epoch", 0)),
        metrics=dict(payload.get("metrics", {})),
        history=list(payload.get("history", [])),
        swath_version=str(payload.get("swath_version", "unknown")),
        format_version=format_version,
        notes=str(payload.get("notes", "")),
    )
    return model, meta


def describe(path: str | Path) -> str:
    """One-paragraph human summary of a checkpoint, for the command line."""
    model, meta = load_checkpoint(path)
    lines = [
        f"{Path(path).name}",
        f"  task          {meta.task.name} — {meta.task.title}",
        f"  classes       {', '.join(meta.task.classes)}",
        f"  bands         {meta.task.in_channels}",
        f"  parameters    {model.num_parameters() / 1e6:.2f}M",
        f"  epoch         {meta.epoch}",
        f"  written by    swath {meta.swath_version}",
    ]
    if meta.metrics:
        scores = "  ".join(f"{key} {value:.4f}" for key, value in sorted(meta.metrics.items()))
        lines.append(f"  metrics       {scores}")
    if meta.notes:
        lines.append(f"  notes         {meta.notes}")
    return "\n".join(lines)
