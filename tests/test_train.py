"""An end-to-end training run on the synthetic tiles.

The point is not accuracy but wiring: that a run produces checkpoints, a history
file and metrics, and that the loss actually goes down on a problem this easy. If
the label pipeline, the loss and the optimiser step were not consistent with each
other, a three-colour task would not be learnable at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from swath.checkpoints import load_checkpoint
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


def test_resume_continues_from_the_stored_epoch(tmp_path: Path, samples, task):
    first = _trainer(tmp_path, samples, task, epochs=2)
    first.fit()

    second = _trainer(tmp_path, samples, task, epochs=4, resume=tmp_path / "run" / "last.pt")
    assert second.start_epoch == 2
    assert len(second.history) == 2

    history = second.fit()["history"]
    assert len(history) == 4
    assert [record["epoch"] for record in history] == [1, 2, 3, 4]


def test_resume_restores_the_optimiser_state(tmp_path: Path, samples, task):
    first = _trainer(tmp_path, samples, task, epochs=2)
    first.fit()

    second = _trainer(tmp_path, samples, task, epochs=4, resume=tmp_path / "run" / "last.pt")
    before = first.optimizer.state_dict()["state"]
    after = second.optimizer.state_dict()["state"]
    assert set(before) == set(after)
    assert before, "AdamW should have moment estimates after two epochs"
    for key in before:
        assert torch.allclose(before[key]["exp_avg"], after[key]["exp_avg"])


def test_resume_carries_the_best_score(tmp_path: Path, samples, task):
    first = _trainer(tmp_path, samples, task, epochs=2)
    result = first.fit()

    second = _trainer(tmp_path, samples, task, epochs=4, resume=tmp_path / "run" / "last.pt")
    assert second.best_score == pytest.approx(result["best_mean_iou"], abs=1e-6)


def test_resume_past_the_epoch_budget_does_nothing(tmp_path: Path, samples, task):
    _trainer(tmp_path, samples, task, epochs=2).fit()
    again = _trainer(tmp_path, samples, task, epochs=2, resume=tmp_path / "run" / "last.pt")
    result = again.fit()
    assert len(result["history"]) == 2


def test_best_checkpoint_stays_weights_only(tmp_path: Path, samples, task):
    """best.pt is the artefact people publish; it should not carry optimiser state."""
    _trainer(tmp_path, samples, task, epochs=2).fit()

    _, best = load_checkpoint(tmp_path / "run" / "best.pt")
    _, last = load_checkpoint(tmp_path / "run" / "last.pt")
    assert not best.resumable
    assert last.resumable
    assert (tmp_path / "run" / "best.pt").stat().st_size < (
        tmp_path / "run" / "last.pt"
    ).stat().st_size


def test_best_checkpoint_carries_the_epoch_it_was_saved_at(tmp_path: Path, samples, task):
    """A checkpoint whose history stops one epoch short of its own is confusing."""
    trainer = _trainer(tmp_path, samples, task, epochs=3)
    trainer.fit()

    _, best = load_checkpoint(tmp_path / "run" / "best.pt")
    assert best.history, "the checkpoint should carry the run so far"
    assert best.history[-1]["epoch"] == best.epoch

    _, last = load_checkpoint(tmp_path / "run" / "last.pt")
    assert last.history[-1]["epoch"] == last.epoch == 3
