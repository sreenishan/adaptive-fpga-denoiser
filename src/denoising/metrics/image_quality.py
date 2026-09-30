"""Image quality metrics (spec section 17).

Three numbers, all comparing an output against a **clean reference**:

``MSE``  mean squared error, 0 for identical images, unbounded above.
``PSNR`` peak signal-to-noise ratio in dB, higher is better, infinite when the
         images are identical.
``SSIM`` structural similarity in [-1, 1], 1 for identical images.

Two rules here exist to stop a number being reported that was never measured:

- **Identical images give infinite PSNR**, not a large finite stand-in. The
  formula divides by MSE; at MSE = 0 the answer is unbounded, and rounding that
  to "100 dB" would put a measurement-shaped value where there is no
  measurement. Callers that must print it render the infinity as such.
- **Every metric needs a reference.** There is no no-reference quality estimate
  in this project, so a denoised image with no clean original has no PSNR. The
  pipeline returns ``None`` in that case rather than comparing the output to
  its own input, which measures how much the filter changed and says nothing at
  all about quality.

SSIM comes from scikit-image rather than a local reimplementation: it has
window, stabiliser and boundary conventions that are easy to get subtly wrong
and impossible to notice afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from skimage.metrics import structural_similarity

from ..noise._common import GrayImage, validate_image

__all__ = [
    "DEFAULT_MAX_PIXEL",
    "ImageQuality",
    "calculate_mse",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_quality",
]

#: Peak value for 8-bit images.
DEFAULT_MAX_PIXEL: Final[int] = 255


@dataclass(frozen=True)
class ImageQuality:
    """The three metrics for one comparison.

    Attributes:
        mse: Mean squared error.
        psnr: Peak signal-to-noise ratio in dB; ``math.inf`` when identical.
        ssim: Structural similarity.
    """

    mse: float
    psnr: float
    ssim: float

    @property
    def identical(self) -> bool:
        """True when the two images were pixel-for-pixel the same."""
        return self.mse == 0.0


def _validate_pair(reference: GrayImage, output: GrayImage) -> tuple[GrayImage, GrayImage]:
    reference = validate_image(reference, "reference")
    output = validate_image(output, "output")
    if reference.shape != output.shape:
        raise ValueError(
            f"reference and output must have the same shape, got "
            f"{reference.shape} and {output.shape}"
        )
    return reference, output


def calculate_mse(reference: GrayImage, output: GrayImage) -> float:
    """Mean squared error between *reference* and *output*.

    Computed in float64 after widening: the difference of two uint8 arrays
    wraps around, so ``10 - 250`` would come out as 16 rather than -240.

    Raises:
        ValueError: if the images differ in shape, or are not 2-D uint8.
    """
    reference, output = _validate_pair(reference, output)
    difference = reference.astype(np.float64) - output.astype(np.float64)
    return float(np.mean(difference**2))


def calculate_psnr(
    reference: GrayImage, output: GrayImage, max_pixel: int = DEFAULT_MAX_PIXEL
) -> float:
    """Peak signal-to-noise ratio in dB.

    Returns:
        ``math.inf`` when the images are identical — see the module docstring.

    Raises:
        ValueError: if the images differ in shape, or *max_pixel* is not > 0.
    """
    if max_pixel <= 0:
        raise ValueError(f"max_pixel must be > 0, got {max_pixel}")
    mse = calculate_mse(reference, output)
    if mse == 0.0:
        return math.inf
    return float(10.0 * math.log10((max_pixel**2) / mse))


def calculate_ssim(
    reference: GrayImage, output: GrayImage, max_pixel: int = DEFAULT_MAX_PIXEL
) -> float:
    """Structural similarity index, via scikit-image.

    Raises:
        ValueError: if the images differ in shape, or are smaller than the
            7x7 window scikit-image needs.
    """
    reference, output = _validate_pair(reference, output)
    if min(reference.shape) < 7:
        raise ValueError(
            f"SSIM needs at least a 7x7 image, got {reference.shape}"
        )
    return float(
        structural_similarity(reference, output, data_range=float(max_pixel))
    )


def calculate_quality(
    reference: GrayImage, output: GrayImage, max_pixel: int = DEFAULT_MAX_PIXEL
) -> ImageQuality:
    """All three metrics for one comparison.

    Args:
        reference: The clean image.
        output: The image being assessed.
        max_pixel: Peak value, 255 for uint8.

    Returns:
        An :class:`ImageQuality`.
    """
    return ImageQuality(
        mse=calculate_mse(reference, output),
        psnr=calculate_psnr(reference, output, max_pixel),
        ssim=calculate_ssim(reference, output, max_pixel),
    )
