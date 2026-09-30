"""Trained-model inference wrapper (spec phase 8).

Implements the :class:`denoising.pipeline.NoiseClassifier` Protocol so the
trained CNN drops straight into ``process_image(classifier=...)``.

Usage::

    from denoising.model.inference import load_classifier
    from denoising.pipeline import process_image

    clf = load_classifier("models/checkpoints/best_model.pt")
    result = process_image(image, inference_config, classifier=clf)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import CLASSES, InferenceConfig
from ..preprocessing import to_model_input
from .cnn import NoiseClassifierCNN

__all__ = ["TrainedClassifier", "load_classifier"]


class TrainedClassifier:
    """Wraps a loaded checkpoint and exposes the ``NoiseClassifier`` protocol.

    Args:
        model: The loaded ``NoiseClassifierCNN``.
        classes: Class names in the order the model was trained on (must match
            ``CLASSES``).
        device: The torch device the model lives on.
        config: Inference configuration, used for image geometry.
    """

    def __init__(
        self,
        model: NoiseClassifierCNN,
        classes: list[str],
        device: torch.device,
        config: InferenceConfig,
    ) -> None:
        self._model = model
        self._classes = classes
        self._device = device
        self._config = config

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """Return ``(noise_class, confidence)`` for one grayscale image.

        The image may be any size — it is resized to the configured geometry
        before inference.  ``confidence`` is the softmax probability of the
        predicted class.
        """
        from ..config import ImageConfig

        img_cfg = ImageConfig(
            self._config.image.width, self._config.image.height, True
        )
        tensor_np = to_model_input(image, img_cfg, add_batch=True)  # (1,1,H,W)
        x = torch.from_numpy(tensor_np).to(self._device)

        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1)[0]  # (num_classes,)

        best_idx = int(probs.argmax().item())
        confidence = float(probs[best_idx].item())
        noise_class = self._classes[best_idx]
        return noise_class, confidence

    def predict_all(self, image: np.ndarray) -> dict[str, float]:
        """Return probabilities for every class, keyed by class name."""
        from ..config import ImageConfig

        img_cfg = ImageConfig(
            self._config.image.width, self._config.image.height, True
        )
        tensor_np = to_model_input(image, img_cfg, add_batch=True)
        x = torch.from_numpy(tensor_np).to(self._device)

        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1)[0]

        return {cls: float(probs[i].item()) for i, cls in enumerate(self._classes)}


def load_classifier(
    checkpoint_path: str | Path,
    config: InferenceConfig,
    *,
    device: str = "auto",
) -> TrainedClassifier:
    """Load a trained checkpoint and return a ready-to-use classifier.

    Args:
        checkpoint_path: Path to ``best_model.pt`` saved by the training loop.
        config: Inference configuration (geometry + thresholds).
        device: ``"auto"`` selects CUDA when available, otherwise CPU.

    Raises:
        FileNotFoundError: if the checkpoint does not exist.
        KeyError: if the checkpoint is missing required keys (wrong file).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Train a model first: python scripts/train.py --dataset data/processed"
        )

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    ckpt = torch.load(path, map_location=dev, weights_only=True)

    required = {"model_state_dict", "classes", "model_config"}
    missing = required - set(ckpt.keys())
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {missing}")

    classes: list[str] = ckpt["classes"]
    if classes != list(CLASSES):
        raise ValueError(
            f"Checkpoint class order {classes} does not match "
            f"CLASSES {list(CLASSES)} — the model was trained on a different label set"
        )

    mc = ckpt["model_config"]
    model = NoiseClassifierCNN(
        num_classes=mc["num_classes"],
        input_channels=mc["input_channels"],
        base_channels=mc["base_channels"],
        dropout=mc.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()

    return TrainedClassifier(model, classes, dev, config)
