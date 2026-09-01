"""Neighbourhood extraction shared by every filter.

This module is the software half of a contract with the hardware. The RTL
window generator (``rtl/common/window_3x3.sv``, phase 17) must produce exactly
these neighbourhoods, including at the image border, or the golden-reference
comparison in phase 24 measures the difference between two boundary policies
and reports it as an RTL bug.

The policy is **edge replication**: the pixel outside the image takes the value
of the nearest pixel inside it. ``configs/hardware.yaml`` allows nothing else,
and a test asserts the software configuration agrees with it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..noise._common import GrayImage, validate_image

__all__ = ["pad_image", "sliding_windows", "validate_kernel_size"]


def validate_kernel_size(kernel_size: int, image: GrayImage | None = None) -> int:
    """Check that *kernel_size* is odd and at least 3.

    Raises:
        TypeError: if *kernel_size* is not an integer.
        ValueError: if it is even, smaller than 3, or larger than the image.
    """
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, (int, np.integer)):
        raise TypeError(f"kernel_size must be an integer, got {type(kernel_size).__name__}")
    kernel_size = int(kernel_size)
    if kernel_size < 3:
        raise ValueError(f"kernel_size must be >= 3, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if image is not None and kernel_size > min(image.shape):
        raise ValueError(
            f"kernel_size {kernel_size} is larger than the image {image.shape}"
        )
    return kernel_size


def pad_image(image: GrayImage, radius: int) -> GrayImage:
    """Pad *image* by *radius* pixels on every side, replicating the edges.

    Args:
        image: 2-D uint8 image.
        radius: Half the kernel size, i.e. 1 for a 3x3 window.

    Returns:
        A new array of shape ``(h + 2r, w + 2r)``.
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if radius == 0:
        return image.copy()
    return np.pad(image, radius, mode="edge")


def sliding_windows(image: GrayImage, kernel_size: int) -> npt.NDArray[np.uint8]:
    """Every ``kernel_size`` x ``kernel_size`` neighbourhood of *image*.

    Args:
        image: 2-D uint8 image.
        kernel_size: Odd window size, >= 3.

    Returns:
        A ``(height, width, kernel_size, kernel_size)`` array whose ``[y, x]``
        entry is the window centred on pixel ``(y, x)``, with replicated edges.
        It is a read-only view over a padded copy, so nothing is duplicated per
        pixel.
    """
    image = validate_image(image)
    kernel_size = validate_kernel_size(kernel_size)
    padded = pad_image(image, kernel_size // 2)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    return windows
