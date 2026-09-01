"""Lightweight CNN noise classifier (spec phase 6).

Architecture: repeated Conv→BN→ReLU blocks with max-pool downsampling, then
global average pooling and a linear head.  Four output logits map to CLASSES =
("clean", "salt_pepper", "gaussian", "speckle").

Design decisions:
- base_channels=16 keeps the parameter count low (~25 k) so it trains fast on
  a CPU and fits comfortably on an FPGA's embedded BRAM if ever ported.
- Global average pooling instead of a flattened FC makes the head independent
  of input resolution — a 224×224 training image and a 64×64 test crop go
  through the same weights.
- Batch-norm after every conv stabilises training without needing careful LR
  tuning.
- The model is constructed entirely from the config dataclass so
  `load_training_config()` is the single source of truth for architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from ..config import TrainingConfig

__all__ = ["NoiseClassifierCNN", "build_model", "ModelInfo"]


@dataclass(frozen=True)
class ModelInfo:
    """Parameter counts and architecture summary, for logging."""

    total_params: int
    trainable_params: int
    input_shape: tuple[int, ...]
    num_classes: int
    base_channels: int

    def __str__(self) -> str:
        return (
            f"NoiseClassifierCNN  classes={self.num_classes}  "
            f"base_channels={self.base_channels}  "
            f"params={self.trainable_params:,}"
        )


class _ConvBlock(nn.Module):
    """Conv 3×3 → BatchNorm → ReLU, with optional max-pool."""

    def __init__(self, in_ch: int, out_ch: int, *, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NoiseClassifierCNN(nn.Module):
    """4-class noise classifier.

    Input:  (B, 1, H, W) float32 in [0, 1]
    Output: (B, num_classes) logits — pass through softmax for probabilities.
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 1,
        base_channels: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            _ConvBlock(input_channels, c, pool=True),    # → c  × H/2  × W/2
            _ConvBlock(c, c * 2, pool=True),             # → 2c × H/4  × W/4
            _ConvBlock(c * 2, c * 4, pool=True),         # → 4c × H/8  × W/8
            _ConvBlock(c * 4, c * 4, pool=False),        # → 4c × H/8  × W/8  (no pool)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)              # → 4c × 1 × 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax over logits)."""
        with torch.no_grad():
            return torch.softmax(self(x), dim=1)


def build_model(config: TrainingConfig) -> NoiseClassifierCNN:
    """Construct the model from a training config."""
    m = config.model
    return NoiseClassifierCNN(
        num_classes=m.num_classes,
        input_channels=m.input_channels,
        base_channels=m.base_channels,
        dropout=m.dropout,
    )


def model_info(model: NoiseClassifierCNN, input_shape: Sequence[int]) -> ModelInfo:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg = model.classifier[-1]  # Linear layer
    return ModelInfo(
        total_params=total,
        trainable_params=trainable,
        input_shape=tuple(input_shape),
        num_classes=cfg.out_features,
        base_channels=model.features[0].block[0].out_channels,
    )
