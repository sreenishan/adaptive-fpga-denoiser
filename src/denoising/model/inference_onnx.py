"""ONNX Runtime inference wrapper — lightweight fallback when best_model.pt is absent.

Implements the same predict/predict_all interface as TrainedClassifier so it
drops straight into process_image(classifier=...) without changes.

On Streamlit Cloud, best_model.pt (~2 GB) is excluded from git.  This module
runs hw_model_pruned.onnx (~265 KB total) via onnxruntime instead, giving
~90% test accuracy with no PyTorch dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import CLASSES, InferenceConfig
from ..preprocessing import to_model_input

__all__ = ["OnnxClassifier", "load_onnx_classifier"]


class OnnxClassifier:
    """Wraps an ONNX Runtime session with the NoiseClassifier protocol."""

    backend: str = "onnx"

    def __init__(self, session, classes: list[str], config: InferenceConfig) -> None:
        self._session = session
        self._classes = classes
        self._config = config
        self._input_name = session.get_inputs()[0].name

    def _run(self, image: np.ndarray) -> np.ndarray:
        from ..config import ImageConfig
        img_cfg = ImageConfig(
            self._config.image.width, self._config.image.height, True
        )
        x = to_model_input(image, img_cfg, add_batch=True).astype(np.float32)
        logits = self._session.run(None, {self._input_name: x})[0][0]
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        probs = self._run(image)
        best = int(probs.argmax())
        return self._classes[best], float(probs[best])

    def predict_all(self, image: np.ndarray) -> dict[str, float]:
        probs = self._run(image)
        return {cls: float(probs[i]) for i, cls in enumerate(self._classes)}


def load_onnx_classifier(
    onnx_path: str | Path,
    config: InferenceConfig,
) -> OnnxClassifier:
    """Load an ONNX model and return a ready-to-use classifier.

    The external-data file (``*.onnx.data``) must be in the same directory.

    Raises:
        FileNotFoundError: if the ONNX file does not exist.
        ImportError: if onnxruntime is not installed.
    """
    import onnxruntime as ort  # deferred — optional dep

    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # suppress INFO/WARNING noise
    session = ort.InferenceSession(str(path), opts)
    return OnnxClassifier(session, list(CLASSES), config)
