"""DnCNN denoising network (17-layer feed-forward residual CNN).

Architecture
------------
Layer  1     : Conv(in_ch, 64, 3, padding=1) → ReLU
Layers 2–16  : Conv(64, 64, 3, padding=1) → BatchNorm(64) → ReLU
Layer  17    : Conv(64, in_ch, 3, padding=1)          (no activation)

Forward pass (residual learning)::

    residual = net(noisy)          # predicted noise
    denoised = clamp(noisy - residual, 0, 1)

The network learns to predict the noise component rather than the clean image
directly.  This converges faster and gives better-conditioned gradients because
the residual (noise) is small relative to the image.

The default depth of 17 follows the original paper; any odd depth ≥ 3 is valid
(enforced by :class:`~denoising.config.DnCNNModelConfig`).
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

__all__ = ["DnCNN", "build_dncnn"]

_MIN_DEPTH: Final[int] = 3


class DnCNN(nn.Module):
    """Feed-forward denoising CNN with residual learning.

    Args:
        depth:          Total number of conv layers (default 17, must be odd).
        filters:        Channels in every hidden layer (default 64).
        input_channels: Input/output channels — 1 for grayscale (default).
    """

    def __init__(
        self,
        depth: int = 17,
        filters: int = 64,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        if depth < _MIN_DEPTH:
            raise ValueError(f"depth must be >= {_MIN_DEPTH}, got {depth}")

        layers: list[nn.Module] = []

        # First layer: Conv → ReLU
        layers.append(nn.Conv2d(input_channels, filters, 3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers: Conv → BN → ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(filters, filters, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv (no activation, no BN)
        layers.append(nn.Conv2d(filters, input_channels, 3, padding=1, bias=True))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Denoise *noisy* by predicting and subtracting the noise component.

        Args:
            noisy: Float tensor in [0, 1], shape (B, C, H, W).

        Returns:
            Denoised tensor in [0, 1], same shape as input.
        """
        residual = self.net(noisy)
        return torch.clamp(noisy - residual, 0.0, 1.0)

    def predict_noise(self, noisy: torch.Tensor) -> torch.Tensor:
        """Return the predicted noise component (before subtraction).

        Useful for inspecting what the network has learned; the denoised
        output is ``clamp(noisy - predict_noise(noisy), 0, 1)``.
        """
        return self.net(noisy)

    @property
    def depth(self) -> int:
        conv_layers = sum(1 for m in self.modules() if isinstance(m, nn.Conv2d))
        return conv_layers

    @property
    def filters(self) -> int:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels > 1:
                return m.out_channels
        return 0


def build_dncnn(
    depth: int = 17,
    filters: int = 64,
    input_channels: int = 1,
    *,
    device: str | torch.device | None = None,
) -> DnCNN:
    """Construct a :class:`DnCNN` and move it to *device*.

    Args:
        depth:          Number of conv layers.
        filters:        Hidden channels.
        input_channels: 1 for grayscale.
        device:         Target device; ``None`` keeps the default (CPU).

    Returns:
        Freshly initialised :class:`DnCNN` on the requested device.
    """
    model = DnCNN(depth=depth, filters=filters, input_channels=input_channels)
    if device is not None:
        model = model.to(device)
    return model
