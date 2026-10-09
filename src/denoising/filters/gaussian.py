"""Gaussian smoothing filter (spec section 14.2).

Two kernels live here, and which one is used matters for hardware:

**The integer binomial kernel** (the default for 3x3, and the one the RTL
implements)::

    1 2 1
    2 4 2   / 16
    1 2 1

Every operation is integer: the weighted sum of nine uint8 pixels is at most
``255 * 16 = 4080`` so a 13-bit accumulator holds it, the weights are powers of
two and their sums, and the division is a shift. The rounding rule is
**round half up**, ``(accumulator + 8) >> 4`` — one adder in hardware and one
of the two obvious choices, so it is written down here rather than left to
whichever language rounds which way. That makes bit-exact agreement with the
RTL achievable, which is what ``max_abs_error.gaussian = 0`` demands.

**A kernel sampled from a Gaussian** for an explicit ``sigma``. This is the
textbook filter and is available for comparison, but it is float arithmetic and
the RTL does not implement it.

The two are not the same filter. The binomial kernel corresponds to a Gaussian
of sigma ~0.85, not the ``sigma: 1.0`` in ``configs/inference.yaml`` — which is
why the config carries an explicit ``integer_kernel`` switch instead of letting
a sigma silently select a kernel nobody built in hardware.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..noise._common import GrayImage, validate_image, validate_non_negative
from ._window import sliding_windows, validate_kernel_size

__all__ = [
    "BINOMIAL_KERNEL_3X3",
    "BINOMIAL_DIVISOR_3X3",
    "BINOMIAL_EFFECTIVE_SIGMA",
    "gaussian_filter",
    "gaussian_kernel",
]

#: The fixed 3x3 kernel the RTL implements. Integer weights, sum 16.
BINOMIAL_KERNEL_3X3: npt.NDArray[np.int32] = np.array(
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.int32
)

#: Sum of :data:`BINOMIAL_KERNEL_3X3`; the division is a 4-bit right shift.
BINOMIAL_DIVISOR_3X3: int = 16

#: The sigma the binomial kernel actually corresponds to, to two decimals.
#: Stated so nobody reads ``sigma: 1.0`` in the config and assumes this kernel
#: honours it.
BINOMIAL_EFFECTIVE_SIGMA: float = 0.85


def gaussian_kernel(kernel_size: int, sigma: float) -> npt.NDArray[np.float64]:
    """A normalised 2-D Gaussian kernel sampled on the pixel grid.

    Args:
        kernel_size: Odd size, >= 3.
        sigma: Standard deviation in pixels, > 0.

    Returns:
        A ``(kernel_size, kernel_size)`` float array summing to 1.
    """
    kernel_size = validate_kernel_size(kernel_size)
    sigma = validate_non_negative(sigma, "sigma")
    if sigma == 0.0:
        raise ValueError("sigma must be > 0 for a sampled Gaussian kernel")
    radius = kernel_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    line = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel = np.outer(line, line)
    return kernel / kernel.sum()


def gaussian_filter(
    image: GrayImage,
    kernel_size: int = 3,
    sigma: float | None = None,
    *,
    integer_kernel: bool = True,
) -> GrayImage:
    """Smooth *image* with a Gaussian-shaped kernel.

    Args:
        image: 2-D uint8 grayscale image. Not modified.
        kernel_size: Odd window size, >= 3.
        sigma: Standard deviation for the sampled kernel. Ignored when
            *integer_kernel* is true, which is the point of the switch being
            explicit.
        integer_kernel: Use the fixed integer binomial kernel — exact, and the
            one the RTL implements. Only defined for ``kernel_size == 3``.

    Returns:
        A new uint8 array with the same shape as *image*.

    Raises:
        ValueError: if *image* is not 2-D uint8, if *kernel_size* is even or
            out of range, if *integer_kernel* is asked for at a size other than
            3, or if the sampled kernel is asked for without a *sigma*.
    """
    image = validate_image(image)
    kernel_size = validate_kernel_size(kernel_size, image)

    if integer_kernel:
        if kernel_size != 3:
            raise ValueError(
                "the integer binomial kernel is only defined for kernel_size=3; "
                "pass integer_kernel=False with a sigma for other sizes"
            )
        windows = sliding_windows(image, 3).astype(np.int32)
        accumulator = (windows * BINOMIAL_KERNEL_3X3).sum(axis=(-2, -1))
        # Round half up, then shift: exactly what the RTL will do.
        return ((accumulator + BINOMIAL_DIVISOR_3X3 // 2) >> 4).astype(np.uint8)

    if sigma is None:
        raise ValueError("sigma is required when integer_kernel is False")
    kernel = gaussian_kernel(kernel_size, sigma)
    windows = sliding_windows(image, kernel_size).astype(np.float64)
    accumulator = (windows * kernel).sum(axis=(-2, -1))
    return np.rint(np.clip(accumulator, 0.0, 255.0)).astype(np.uint8)
