"""An end-to-end training run on the synthetic tiles.

The point is not accuracy but wiring: that a run produces checkpoints, a history
file and metrics, and that the loss actually goes down on a problem this easy. If
the label pipeline, the loss and the optimiser step were not consistent with each
other, a three-colour task would not be learnable at all.
"""

from __future__ import annotations

from pathlib import Path

from swath.data.dataset import SegmentationDataset
from swath.data.transforms import build_eval_transform, build_train_transform
from swath.train import TrainConfig, Trainer, learning_rate_at


def _trainer(tmp_path: Path, samples, task, **overrides) -> Trainer:
    config = TrainConfig(
        output_dir=tmp_path / "run",
        epochs=overrides.pop("epochs", 2),
        batch_size=2,
        crop_size=32,
        val_crop_size=64,
        learning_rate=3e-3,
        base_channels=8,
        depth=2,
        blocks_per_stage=1,
        num_workers=0,
        warmup_epochs=0,
        device="cpu",
        amp=False,
        **overrides,
    )
    train_dataset = SegmentationDataset(samples, task, transform=build_train_transform(32))
    val_dataset = SegmentationDataset(samples[:4], task, transform=build_eval_transform(64))
    return Trainer(task, train_dataset, val_dataset, config)


def test_run_produces_checkpoints_and_history(tmp_path: Path, samples, task):
    trainer = _trainer(tmp_path, samples, task)
    result = trainer.fit()

    run = tmp_path / "run"
    assert (run / "best.pt").is_file()
    assert (run / "last.pt").is_file()
    assert (run / "history.csv").is_file()
    assert (run / "config.json").is_file()
    assert (run / "metrics.txt").is_file()
    assert len(result["history"]) == 2
    assert 0.0 <= result["best_mean_iou"] <= 1.0


def test_loss_decreases_on_a_learnable_task(tmp_path: Path, samples, task):
    trainer = _trainer(tmp_path, samples, task, epochs=6)
    history = trainer.fit()["history"]
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_history_columns_include_per_class_iou(tmp_path: Path, samples, task):
    trainer = _trainer(tmp_path, samples, task)
    trainer.fit()
    header = (tmp_path / "run" / "history.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "val_miou" in header
    assert "iou_stripe" in header


def test_class_weights_are_derived_from_the_data(tmp_path: Path, samples, task):
    trainer = _trainer(tmp_path, samples, task)
    weights = trainer.criterion.cross_entropy.weight
    assert weights is not None
    assert weights.shape == (task.num_classes,)
    # Background dominates these tiles, so it must be down-weighted.
    assert float(weights[0]) < float(weights[1])


def test_schedule_warms_up_then_decays():
    assert learning_rate_at(0, 100, 10, 0.02) < learning_rate_at(9, 100, 10, 0.02)
    assert learning_rate_at(10, 100, 10, 0.02) == 1.0
    assert learning_rate_at(99, 100, 10, 0.02) < 0.1
    assert learning_rate_at(99, 100, 10, 0.02) >= 0.02
