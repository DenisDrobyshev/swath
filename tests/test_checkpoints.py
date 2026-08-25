from __future__ import annotations

from pathlib import Path

import pytest
import torch

from swath.checkpoints import describe, load_checkpoint, model_config, save_checkpoint
from swath.models import build_model


def test_round_trip_preserves_weights_and_task(tmp_path: Path, tiny_model, task):
    path = save_checkpoint(
        tmp_path / "model.pt", tiny_model, task, epoch=7, metrics={"mean_iou": 0.42}
    )
    restored, meta = load_checkpoint(path)

    assert meta.task == task
    assert meta.epoch == 7
    assert meta.metrics["mean_iou"] == 0.42
    for before, after in zip(
        tiny_model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        assert torch.equal(before, after)


def test_restored_model_predicts_identically(tmp_path: Path, tiny_model, task):
    tiny_model.eval()
    batch = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = tiny_model(batch)

    path = save_checkpoint(tmp_path / "model.pt", tiny_model, task)
    restored, _ = load_checkpoint(path)
    with torch.no_grad():
        actual = restored(batch)

    assert torch.allclose(expected, actual, atol=1e-6)


def test_model_config_reconstructs_the_architecture():
    model = build_model(
        in_channels=5,
        num_classes=9,
        base_channels=16,
        depth=3,
        blocks_per_stage=2,
        norm="group",
        dropout=0.2,
    )
    config = model_config(model)
    assert config == {
        "in_channels": 5,
        "num_classes": 9,
        "base_channels": 16,
        "depth": 3,
        "blocks_per_stage": 2,
        "norm": "group",
        "dropout": 0.2,
    }
    rebuilt = build_model(**config)
    assert rebuilt.num_parameters() == model.num_parameters()


def test_checkpoint_is_self_describing(checkpoint: Path):
    _, meta = load_checkpoint(checkpoint)
    # The palette and class names travel with the weights, so a service can
    # colour the output without consulting the registry.
    assert meta.task.classes == ("background", "stripe", "blob")
    assert meta.task.palette[1] == (220, 40, 40)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "absent.pt")


def test_newer_format_is_refused(tmp_path: Path, tiny_model, task):
    path = save_checkpoint(tmp_path / "model.pt", tiny_model, task)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="newer swath"):
        load_checkpoint(path)


def test_describe_mentions_the_task_and_metrics(checkpoint: Path):
    text = describe(checkpoint)
    assert "test" in text
    assert "mean_iou" in text
    assert "synthetic fixture" in text
