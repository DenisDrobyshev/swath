"""Loss functions for dense classification.

Land cover is badly imbalanced: on a typical aerial tile, background and
agriculture cover most of the pixels while roads cover a few percent. Plain
cross-entropy optimises the frequent classes and quietly gives up on the rest,
so the default here is cross-entropy plus a soft Dice term, which is computed
per class and therefore weighs a thin road as heavily as a large field.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from swath.tasks import IGNORE_INDEX


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice, averaged over the classes present in the batch."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probabilities = logits.softmax(dim=1)

        valid = (target != self.ignore_index).unsqueeze(1)
        # One-hot needs in-range values, so ignored pixels are parked on class 0
        # and then masked out; they contribute to neither intersection nor union.
        safe_target = target.masked_fill(target == self.ignore_index, 0)
        one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).to(probabilities.dtype)

        probabilities = probabilities * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        cardinality = probabilities.sum(dims) + one_hot.sum(dims)

        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        present = one_hot.sum(dims) > 0
        if not bool(present.any()):
            return logits.sum() * 0.0
        return 1.0 - dice[present].mean()


class SegmentationLoss(nn.Module):
    """Weighted sum of cross-entropy and soft Dice.

    Args:
        weight: Optional per-class weights for the cross-entropy term.
        dice_weight: Contribution of the Dice term; ``0`` disables it.
        label_smoothing: Cross-entropy label smoothing.
        ignore_index: Label value excluded from the loss.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.0,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing
        )
        self.dice = SoftDiceLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.cross_entropy(logits, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, target)
        return loss


def class_weights_from_frequency(
    counts: torch.Tensor, mode: str = "sqrt_inverse", clamp: float = 10.0
) -> torch.Tensor:
    """Turn pixel counts per class into cross-entropy weights.

    ``sqrt_inverse`` is the middle ground that works on most land-cover data:
    plain inverse frequency over-corrects and makes training unstable when one
    class is a hundred times rarer than another.
    """
    counts = counts.to(torch.float64).clamp(min=1.0)
    frequency = counts / counts.sum()
    if mode == "inverse":
        weights = 1.0 / frequency
    elif mode == "sqrt_inverse":
        weights = 1.0 / frequency.sqrt()
    elif mode == "none":
        weights = torch.ones_like(frequency)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    weights = weights / weights.mean()
    return weights.clamp(max=clamp).to(torch.float32)
