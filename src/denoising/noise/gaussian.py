"""Additive Gaussian noise (spec section 7.2).

Every pixel is perturbed by an independent draw from ``N(mean, sigma^2)``:

```text
out = clip(in + N(mean, sigma^2), 0, 1)
```

with the arithmetic done in normalised [0, 1] units, so ``sigma = 0.08`` means
8% of full scale (about 20 grey levels) regardless of the pixel depth.
"""

from __future__ import annotations

from ._common import (
    GrayImage,
    SeedLike,
    resolve_rng,
    to_float01,
    to_uint8,
    validate_finite,
    validate_image,
    validate_non_negative,
)

__all__ = ["add_gaussian_noise"]


def add_gaussian_noise(
    image: GrayImage,
    mean: float = 0.0,
    sigma: float = 0.05,
    seed: SeedLike = None,
) -> GrayImage:
    """Add zero- or shifted-mean Gaussian noise to a grayscale image.

    The result is clipped to the valid pixel range before being rounded back to
    uint8, so a bright pixel saturates at 255 instead of wrapping round to a
    dark one.

    Note that clipping is lossy at high sigma: with ``sigma = 0.10`` a pixel at
    250 has most of its upward noise removed, so the empirical standard
    deviation of a corrupted bright region is below ``sigma * 255``. That is a
    property of 8-bit imaging, not an error, and the same clipping happens in
    any real sensor.

    Args:
        image: 2-D uint8 grayscale image. Not modified.
        mean: Mean of the noise, in normalised units. May be negative.
        sigma: Standard deviation of the noise, in normalised units, >= 0.
            ``sigma = 0`` with ``mean = 0`` returns the image unchanged.
        seed: Int, :class:`numpy.random.Generator`, or ``None`` for fresh
            entropy.

    Returns:
        A new uint8 array with the same shape as *image*.

    Raises:
        TypeError: if *image* is not an array, or a parameter is not a number.
        ValueError: if *image* is not 2-D uint8, or *sigma* is negative.
    """
    image = validate_image(image)
    mean = validate_finite(mean, "mean")
    sigma = validate_non_negative(sigma, "sigma")
    rng = resolve_rng(seed)

    values = to_float01(image)
    if sigma > 0.0:
        values = values + rng.normal(loc=mean, scale=sigma, size=values.shape)
    elif mean != 0.0:
        values = values + mean
    return to_uint8(values)
