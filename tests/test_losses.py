from __future__ import annotations

import torch

from swath.losses import SegmentationLoss, SoftDiceLoss, class_weights_from_frequency
from swath.tasks import IGNORE_INDEX


def _confident_logits(target: torch.Tensor, num_classes: int, magnitude: float = 12.0):
    logits = torch.zeros(1, num_classes, *target.shape[-2:])
    safe = target.clone()
    safe[safe == IGNORE_INDEX] = 0
    return logits.scatter_(1, safe.unsqueeze(1), magnitude)


def test_dice_is_near_zero_for_a_correct_prediction():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = 1
    loss = SoftDiceLoss()(_confident_logits(target, 2), target)
    assert float(loss) < 0.01


def test_dice_is_large_for_an_inverted_prediction():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = 1
    inverted = 1 - target
    loss = SoftDiceLoss()(_confident_logits(inverted, 2), target)
    assert float(loss) > 0.9


def test_ignored_pixels_do_not_contribute():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = 1
    with_ignore = target.clone()
    with_ignore[:, :2] = IGNORE_INDEX

    logits = _confident_logits(target, 2)
    # Corrupt exactly the region that is ignored; the loss must not notice.
    logits[:, :, :2] = torch.randn_like(logits[:, :, :2]) * 5

    baseline = SoftDiceLoss()(_confident_logits(target, 2)[:, :, 2:], target[:, 2:])
    masked = SoftDiceLoss()(logits, with_ignore)
    assert abs(float(masked) - float(baseline)) < 0.05


def test_all_ignored_returns_a_finite_zero():
    target = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.long)
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    loss = SoftDiceLoss()(logits, target)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_combined_loss_falls_when_the_prediction_improves():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = 1
    criterion = SegmentationLoss(dice_weight=0.5)
    good = float(criterion(_confident_logits(target, 2), target))
    bad = float(criterion(torch.zeros(1, 2, 8, 8), target))
    assert good < bad


def test_class_weights_favour_rare_classes():
    counts = torch.tensor([1000, 10])
    weights = class_weights_from_frequency(counts)
    assert weights[1] > weights[0]
    assert float(weights.max()) <= 10.0


def test_class_weight_mode_none_is_uniform():
    weights = class_weights_from_frequency(torch.tensor([1000, 10]), mode="none")
    assert torch.allclose(weights, torch.ones_like(weights))
