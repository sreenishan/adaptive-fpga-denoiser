"""Adaptive Wiener filter (spec section 14.3).

Local-statistics form, computed over the same replicated-edge neighbourhood as
the other two filters::

    out = mean + max(var - noise_var, 0) / max(var, noise_var) * (pixel - mean)

where ``mean`` and ``var`` are the local mean and variance of the window and
``noise_var`` is the noise power — configured, or estimated as the average of
the local variances across the image when it is not.

Reading the equation as behaviour: the fraction is a **gain between 0 and 1**.
Where the window is flat (``var <= noise_var``) the gain is 0 and the output is
the local mean, smoothing hardest. Where the window has structure
(``var >> noise_var``) the gain approaches 1 and the pixel passes through
untouched, preserving edges. That is why it suits speckle, whose strength
scales with the signal, and why a fixed kernel does not.

**Numerical stability.** The denominator is ``max(var, noise_var)``, never
``var`` alone, so it can only be zero when both are zero — and then the
numerator is zero too. A floor of :data:`_EPSILON` makes that case return the
local mean rather than dividing zero by zero, which is the right answer: a flat
window with no noise is its own mean.

This is the software reference. The RTL version (phase 20) must approximate the
division, and ``configs/hardware.yaml`` allows it one grey level of error
(``max_abs_error.wiener = 1``). Whatever approximation is chosen gets documented
and compared against this function — it does not get to quietly become a
different filter.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ..noise._common import GrayImage, validate_image, validate_non_negative
from ._window import sliding_windows, validate_kernel_size

__all__ = ["wiener_filter", "estimate_noise_variance"]

#: Guards the 0/0 case only; far below one grey level squared.
_EPSILON: Final[float] = 1e-12


def _local_statistics(image: GrayImage, kernel_size: int):
    """Local mean and (biased) variance of every neighbourhood."""
    windows = sliding_windows(image, kernel_size).astype(np.float64)
    mean = windows.mean(axis=(-2, -1))
    variance = windows.var(axis=(-2, -1))
    return mean, variance


def estimate_noise_variance(image: GrayImage, kernel_size: int = 3) -> float:
    """Estimate the noise power as the mean of the local variances.

    The standard estimator for this filter. It is an estimate and is reported
    as one: on a heavily textured image it mistakes structure for noise and
    over-smooths, which is a property of the method, not a defect in this code.

    Args:
        image: 2-D uint8 grayscale image.
        kernel_size: Odd window size, >= 3.

    Returns:
        The estimated noise variance, in squared grey levels (0-255 scale).
    """
    image = validate_image(image)
    kernel_size = validate_kernel_size(kernel_size, image)
    _, variance = _local_statistics(image, kernel_size)
    return float(variance.mean())


def wiener_filter(
    image: GrayImage,
    kernel_size: int = 3,
    noise_variance: float | None = None,
) -> GrayImage:
    """Apply the adaptive Wiener filter.

    Args:
        image: 2-D uint8 grayscale image. Not modified.
        kernel_size: Odd window size, >= 3.
        noise_variance: Noise power in squared grey levels (0-255 scale). When
            ``None`` it is estimated with :func:`estimate_noise_variance` —
            ``None`` means "nobody measured this", which is not the same claim
            as 0, and 0 would disable the filter entirely.

    Returns:
        A new uint8 array with the same shape as *image*.

    Raises:
        TypeError: if *image* is not an array or a parameter is not a number.
        ValueError: if *image* is not 2-D uint8, *kernel_size* is even or out
            of range, or *noise_variance* is negative.
    """
    image = validate_image(image)
    kernel_size = validate_kernel_size(kernel_size, image)

    mean, variance = _local_statistics(image, kernel_size)
    if noise_variance is None:
        noise = float(variance.mean())
    else:
        noise = validate_non_negative(noise_variance, "noise_variance")

    numerator = np.maximum(variance - noise, 0.0)
    denominator = np.maximum(np.maximum(variance, noise), _EPSILON)
    gain = numerator / denominator

    values = mean + gain * (image.astype(np.float64) - mean)
    return np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)
