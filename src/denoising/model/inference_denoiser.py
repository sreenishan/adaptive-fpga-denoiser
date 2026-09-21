"""Single-image inference with a saved DnCNN checkpoint.

Keeps the denoising pipeline separate from the classifier pipeline — this
module never imports :class:`~denoising.model.cnn.NoiseClassifierCNN` and the
classifier pipeline never imports :class:`DnCNN`.

Usage::

    inferencer = DnCNNInferencer.from_checkpoint("models/checkpoints/denoiser.pt")
    denoised = inferencer.denoise(noisy_uint8)   # uint8 → uint8
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..noise._common import GrayImage
from .dncnn import DnCNN, build_dncnn

__all__ = ["DnCNNInferencer", "load_denoiser"]


class DnCNNInferencer:
    """Wraps a loaded DnCNN for single-image inference.

    Accepts and returns uint8 numpy arrays (the same type as the rest of the
    pipeline) so callers never need to know about torch tensors.
    """

    def __init__(self, model: DnCNN, device: torch.device) -> None:
        self._model = model.eval()
        self._device = device

    @classmethod
    def from_checkpoint(
        cls, path: Path | str, *, device: str | torch.device | None = None
    ) -> "DnCNNInferencer":
        """Load from a ``denoiser.pt`` checkpoint dict.

        Args:
            path:   Path to the checkpoint saved by :func:`~denoising.model.train_denoiser.train_denoiser`.
            device: Target device; ``None`` → auto (CUDA if available, else CPU).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        ckpt = torch.load(path, map_location=device)
        depth = ckpt.get("depth", 17)
        filters = ckpt.get("filters", 64)
        model = build_dncnn(depth=depth, filters=filters, input_channels=1, device=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return cls(model, device)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "DnCNN",
            "depth": self._model.depth,
            "filters": self._model.filters,
            "device": str(self._device),
        }

    def denoise(self, image: GrayImage) -> GrayImage:
        """Denoise a uint8 grayscale image.

        Args:
            image: 2-D uint8 numpy array, shape (H, W).

        Returns:
            Denoised image, same shape and dtype as input.
        """
        tensor = (
            torch.from_numpy(image.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
        )
        with torch.no_grad():
            out = self._model(tensor)
        result = (out.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return result


def load_denoiser(
    path: Path | str | None = None, *, device: str | torch.device | None = None
) -> DnCNNInferencer:
    """Convenience wrapper around :meth:`DnCNNInferencer.from_checkpoint`.

    Args:
        path:   Checkpoint path; defaults to ``models/checkpoints/denoiser.pt``
                relative to the project root.
        device: Target device.
    """
    if path is None:
        from ..config import PROJECT_ROOT
        path = PROJECT_ROOT / "models" / "checkpoints" / "denoiser.pt"
    return DnCNNInferencer.from_checkpoint(path, device=device)
