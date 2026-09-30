"""Adaptive FPGA-based image noise detection and reduction.

The package is layered so that each phase of the project can be exercised on
its own:

``config``
    Typed loading and validation of ``configs/*.yaml``.
``noise``
    Reproducible salt-and-pepper, Gaussian and speckle generators.
``dataset``
    Dataset generation, splitting and loading.
``model``
    The four-class CNN noise classifier: training, evaluation, inference.
``filters``
    Software reference filters and the centralised filter selector.
``pipeline``
    Preprocess -> classify -> select -> filter, end to end.
``metrics``
    MSE, PSNR and SSIM.
``visualization``
    Plots for the reports under ``results/``.

Only :mod:`denoising.config` and :mod:`denoising.logging_utils` are imported
here, so importing the package never pulls in a machine-learning framework.
"""

from __future__ import annotations

from .config import (
    BOUNDARY_MODES,
    CLASSES,
    FALLBACK_FILTERS,
    PROJECT_ROOT,
    ConfigError,
    DatasetConfig,
    HardwareConfig,
    InferenceConfig,
    TrainingConfig,
    load_dataset_config,
    load_hardware_config,
    load_inference_config,
    load_training_config,
)
from .logging_utils import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BOUNDARY_MODES",
    "CLASSES",
    "FALLBACK_FILTERS",
    "PROJECT_ROOT",
    "ConfigError",
    "DatasetConfig",
    "HardwareConfig",
    "InferenceConfig",
    "TrainingConfig",
    "load_dataset_config",
    "load_hardware_config",
    "load_inference_config",
    "load_training_config",
    "configure_logging",
    "get_logger",
]
