from __future__ import annotations

import pytest
import torch

from swath.models import build_model


@pytest.mark.parametrize("size", [64, 96, 128])
def test_output_matches_input_resolution(size: int):
    model = build_model(num_classes=4, base_channels=8, depth=2, blocks_per_stage=1)
    output = model(torch.randn(1, 3, size, size))
    assert output.shape == (1, 4, size, size)


def test_any_band_count_is_accepted():
    model = build_model(in_channels=10, num_classes=3, base_channels=8, depth=2)
    output = model(torch.randn(2, 10, 64, 64))
    assert output.shape == (2, 3, 64, 64)


def test_non_square_input():
    model = build_model(num_classes=2, base_channels=8, depth=2, blocks_per_stage=1)
    output = model(torch.randn(1, 3, 64, 96))
    assert output.shape == (1, 2, 64, 96)


def test_size_divisor_follows_depth():
    assert build_model(depth=3, base_channels=8).size_divisor == 8
    assert build_model(depth=5, base_channels=8).size_divisor == 32


def test_group_norm_variant_runs_with_batch_of_one():
    model = build_model(num_classes=2, base_channels=8, depth=2, norm="group")
    model.train()
    output = model(torch.randn(1, 3, 64, 64))
    assert torch.isfinite(output).all()


def test_unknown_norm_is_rejected():
    with pytest.raises(ValueError, match="unknown norm"):
        build_model(norm="layer")


def test_gradients_reach_the_stem():
    model = build_model(num_classes=3, base_channels=8, depth=2, blocks_per_stage=1)
    output = model(torch.randn(1, 3, 64, 64))
    output.mean().backward()
    stem_weight = model.stem[0].weight
    assert stem_weight.grad is not None
    assert torch.isfinite(stem_weight.grad).all()
    assert float(stem_weight.grad.abs().sum()) > 0
