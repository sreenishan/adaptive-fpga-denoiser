"""Image quality metrics: MSE, PSNR, SSIM (spec section 17).

Every metric compares an output against a **clean reference**. There is no
no-reference quality estimate here, so an image with no original has no PSNR —
the pipeline reports ``None`` rather than comparing an output to its own input.
"""

from __future__ import annotations

from .image_quality import (
    DEFAULT_MAX_PIXEL,
    ImageQuality,
    calculate_mse,
    calculate_psnr,
    calculate_quality,
    calculate_ssim,
)

__all__ = [
    "DEFAULT_MAX_PIXEL",
    "ImageQuality",
    "calculate_mse",
    "calculate_psnr",
    "calculate_quality",
    "calculate_ssim",
]
