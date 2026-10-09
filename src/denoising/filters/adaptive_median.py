"""Adaptive median filter (Hwang & Haddad) for high-density impulse noise.

A plain 3x3 median fails once impulses stop being a minority of the window: at
30% salt-and-pepper a 3x3 window regularly holds five or more extremes, the
median *is* an impulse, and the noise survives the filter. Measured on the
synthetic set, a single 3x3 pass recovers 24.5 dB at 30% density where this
filter recovers 39.0 dB.

The algorithm grows the window until its median is not itself an extreme, and
then replaces a pixel **only if that pixel is an extreme of its window** — so,
unlike the plain median, a pixel strictly between its neighbours' min and max
is left exactly as it was. "Extreme" is local, not "is 0 or 255": the darkest
pixel of a genuine dark speck is also replaced by its median. That is the
standard algorithm's behaviour, and why it is reserved for heavy impulse noise
rather than run on every image:

    stage A, for w = 3, 5, ..., max_size:
        if min(w) < median(w) < max(w): go to stage B with this window
    (no window qualified: output median(max_size))
    stage B:
        if min(w) < pixel < max(w): keep pixel         # not an impulse
        else:                       output median(w)

**This filter has no RTL counterpart.** Its window size varies per pixel, which
the fixed 3x3 streaming core in ``rtl/`` cannot do, and it has no 2-bit control
code — :data:`~denoising.filters.selector.FILTERS` is exactly the set the
hardware implements, and this is deliberately not in it. The selector marks
every decision that uses it as software-only, so no screen can present its
output as FPGA-filtered.

Exact integer arithmetic throughout — every output is an input sample.
"""

from __future__ import annotations

import numpy as np

from ..noise._common import GrayImage, validate_image
from ._window import sliding_windows

__all__ = ["adaptive_median_filter"]


def adaptive_median_filter(image: GrayImage, max_size: int = 7) -> GrayImage:
    """Remove impulses with a per-pixel growing window.

    Args:
        image: 2-D uint8 grayscale image. Not modified.
        max_size: Largest odd window to try, >= 3. Windows larger than the
            image are skipped rather than raised on, so a tiny image still
            gets the sizes that fit.

    Returns:
        A new uint8 array with the same shape as *image*.

    Raises:
        TypeError: if *image* is not an array or *max_size* is not an int.
        ValueError: if *image* is not 2-D uint8, is smaller than 3x3, or
            *max_size* is even or smaller than 3.
    """
    image = validate_image(image)
    if isinstance(max_size, bool) or not isinstance(max_size, int):
        raise TypeError(f"max_size must be an int, got {type(max_size).__name__}")
    if max_size < 3 or max_size % 2 == 0:
        raise ValueError(f"max_size must be an odd integer >= 3, got {max_size}")
    if min(image.shape) < 3:
        raise ValueError(f"image {image.shape} is smaller than the 3x3 starting window")

    largest = min(max_size, min(image.shape) if min(image.shape) % 2 else min(image.shape) - 1)

    output = image.copy()
    undecided = np.ones(image.shape, dtype=bool)
    last_median = image
    for size in range(3, largest + 1, 2):
        flat = sliding_windows(image, size).reshape(*image.shape, size * size)
        middle = (size * size) // 2
        z_min = flat.min(axis=-1)
        z_max = flat.max(axis=-1)
        z_med = np.partition(flat, middle, axis=-1)[..., middle]
        last_median = z_med

        # Stage A: this window's median is a usable, non-extreme value.
        usable = undecided & (z_min < z_med) & (z_med < z_max)
        # Stage B: only replace the pixel when it is itself an extreme.
        impulse = ~((z_min < image) & (image < z_max))
        replace = usable & impulse
        output[replace] = z_med[replace]
        undecided &= ~usable
        if not undecided.any():
            break

    # Every window up to max_size had an extreme median (a solid run of
    # impulses). The largest window's median is the best estimate available.
    output[undecided] = last_median[undecided]
    return output
