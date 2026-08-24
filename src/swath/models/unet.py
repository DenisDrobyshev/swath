"""A residual U-Net written in plain PyTorch.

The architecture is deliberately dependency-free: no pretrained backbone, no
segmentation library. That costs a little accuracy against an ImageNet-pretrained
encoder, and buys three things that matter more here: the input can have any
number of channels (multispectral imagery is not RGB), the checkpoint is a few
tens of megabytes, and the whole model fits in one readable file.
"""

from __future__ import annotations

import torch
from torch import nn


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        # Eight groups keeps normalisation stable when the batch is tiny, which
        # is the usual situation when tiles are large.
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"unknown norm {kind!r}, expected batch or group")


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a projected identity shortcut."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = _norm(norm, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _norm(norm, out_channels)
        self.act = nn.SiLU(inplace=True)
        self.shortcut: nn.Module = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + identity)


class EncoderStage(nn.Module):
    """Downsample by two, then refine."""

    def __init__(
        self, in_channels: int, out_channels: int, blocks: int, norm: str = "batch"
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        layers = [ResidualBlock(in_channels, out_channels, norm)]
        layers += [ResidualBlock(out_channels, out_channels, norm) for _ in range(blocks - 1)]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.pool(x))


class DecoderStage(nn.Module):
    """Upsample, concatenate the matching skip connection, then refine.

    Bilinear upsampling followed by a convolution is used instead of a
    transposed convolution: it does not produce the checkerboard artefacts that
    are so visible once a mask is drawn on top of the image.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        blocks: int,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(norm, out_channels),
            nn.SiLU(inplace=True),
        )
        layers = [ResidualBlock(out_channels + skip_channels, out_channels, norm)]
        layers += [ResidualBlock(out_channels, out_channels, norm) for _ in range(blocks - 1)]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.blocks(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """Residual U-Net for dense per-pixel classification.

    Args:
        in_channels: Number of bands in the input raster (3 for RGB).
        num_classes: Number of output classes.
        base_channels: Width of the first stage; every stage doubles it.
        depth: Number of downsampling stages.
        blocks_per_stage: Residual blocks in each encoder and decoder stage.
        norm: Either batch or group normalisation.
        dropout: Dropout applied to the bottleneck only.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 48,
        depth: int = 4,
        blocks_per_stage: int = 2,
        norm: str = "batch",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth

        widths = [base_channels * (2**i) for i in range(depth + 1)]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, padding=1, bias=False),
            _norm(norm, widths[0]),
            nn.SiLU(inplace=True),
            ResidualBlock(widths[0], widths[0], norm),
        )
        self.encoder = nn.ModuleList(
            EncoderStage(widths[i], widths[i + 1], blocks_per_stage, norm) for i in range(depth)
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.decoder = nn.ModuleList(
            DecoderStage(widths[i + 1], widths[i], widths[i], blocks_per_stage, norm)
            for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(widths[0], num_classes, 1)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        x = self.stem(x)
        for stage in self.encoder:
            skips.append(x)
            x = stage(x)
        x = self.dropout(x)
        for stage, skip in zip(self.decoder, reversed(skips), strict=True):
            x = stage(x, skip)
        return self.head(x)

    @property
    def size_divisor(self) -> int:
        """Input side lengths must be a multiple of this for exact skip alignment."""
        return 2**self.depth

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_model(
    in_channels: int = 3,
    num_classes: int = 2,
    base_channels: int = 48,
    depth: int = 4,
    blocks_per_stage: int = 2,
    norm: str = "batch",
    dropout: float = 0.1,
) -> UNet:
    """Construct a :class:`UNet` from plain keyword arguments.

    Everything that rebuilds a model from a checkpoint goes through this
    function, so a checkpoint only ever has to store these seven numbers.
    """
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        depth=depth,
        blocks_per_stage=blocks_per_stage,
        norm=norm,
        dropout=dropout,
    )
