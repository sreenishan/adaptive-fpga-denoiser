"""CNN noise classifier — model, training loop and inference wrapper."""

# torch-dependent exports — only available when PyTorch is installed.
try:
    from .cnn import NoiseClassifierCNN, build_model, model_info
    from .inference import TrainedClassifier, load_classifier
    from .optimize import evaluate, model_size_kb, prune_model, quantize_static
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ONNX-based inference — available whenever onnxruntime is installed,
# regardless of whether PyTorch is present.  This is the Streamlit Cloud
# fallback path when best_model.pt is excluded from the repo.
from .inference_onnx import OnnxClassifier, load_onnx_classifier

__all__ = [
    # torch path (conditionally importable)
    "NoiseClassifierCNN",
    "build_model",
    "model_info",
    "TrainedClassifier",
    "load_classifier",
    "prune_model",
    "quantize_static",
    "evaluate",
    "model_size_kb",
    # onnx path (always importable)
    "OnnxClassifier",
    "load_onnx_classifier",
]
