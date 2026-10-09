"""Median filter (spec section 14.1).

The output pixel is the median of its neighbourhood. Because the median of an
odd-sized window is one of the input samples, the result is exact integer
arithmetic — no rounding rule, no fixed-point format — which is why the RTL
median filter is required to be **bit-exact** against this implementation
(``configs/hardware.yaml``, ``max_abs_error.median = 0``).

This is the right filter for salt-and-pepper noise precisely because it selects
rather than averages: an impulse is an extreme value, and extremes never win a
median vote unless they are the majority.
"""

from __future__ import annotations

import numpy as np

from ..noise._common import GrayImage, validate_image
from ._window import sliding_windows, validate_kernel_size

__all__ = ["median_filter"]


def median_filter(image: GrayImage, kernel_size: int = 3) -> GrayImage:
    """Replace every pixel with the median of its neighbourhood.

    Args:
        image: 2-D uint8 grayscale image. Not modified.
        kernel_size: Odd window size, >= 3. Only 3 has an RTL counterpart.

    Returns:
        A new uint8 array with the same shape as *image*.

    Raises:
        TypeError: if *image* is not an array or *kernel_size* is not an int.
        ValueError: if *image* is not 2-D uint8, or *kernel_size* is even,
            smaller than 3, or larger than the image.
    """
    image = validate_image(image)
    kernel_size = validate_kernel_size(kernel_size, image)

    windows = sliding_windows(image, kernel_size)
    flat = windows.reshape(*image.shape, kernel_size * kernel_size)
    middle = (kernel_size * kernel_size) // 2
    # partition is enough: only the middle order statistic is needed, and it is
    # an actual sample, so the result stays uint8 without any rounding.
    return np.partition(flat, middle, axis=-1)[..., middle]
