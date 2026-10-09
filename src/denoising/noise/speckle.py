"""Multiplicative speckle noise (spec section 7.3).

```text
out = clip(in + in * N(0, variance), 0, 1)
```

The perturbation scales with the pixel itself, which is what distinguishes
speckle from additive Gaussian noise and why it needs an adaptive filter rather
than a fixed kernel: bright regions are corrupted hard, dark regions barely at
all, and a black pixel is left exactly black.
"""

from __future__ import annotations

import math

from ._common import (
    GrayImage,
    SeedLike,
    resolve_rng,
    to_float01,
    to_uint8,
    validate_image,
    validate_non_negative,
)

__all__ = ["add_speckle_noise"]


def add_speckle_noise(
    image: GrayImage,
    variance: float = 0.05,
    seed: SeedLike = None,
) -> GrayImage:
    """Apply multiplicative speckle noise to a grayscale image.

    The noise term is drawn from ``N(0, variance)`` — the argument is a
    **variance**, not a standard deviation, matching ``configs/dataset.yaml``
    and the usual formulation of this model. Confusing the two changes the
    intensity by a square root, which looks like a mislabelled dataset rather
    than a bug.

    Args:
        image: 2-D uint8 grayscale image. Not modified.
        variance: Variance of the multiplicative term, >= 0. ``0`` returns the
            image unchanged.
        seed: Int, :class:`numpy.random.Generator`, or ``None`` for fresh
            entropy.

    Returns:
        A new uint8 array with the same shape as *image*.

    Raises:
        TypeError: if *image* is not an array, or *variance* is not a number.
        ValueError: if *image* is not 2-D uint8, or *variance* is negative.
    """
    image = validate_image(image)
    variance = validate_non_negative(variance, "variance")
    rng = resolve_rng(seed)

    values = to_float01(image)
    if variance > 0.0:
        noise = rng.normal(loc=0.0, scale=math.sqrt(variance), size=values.shape)
        values = values + values * noise
    return to_uint8(values)
