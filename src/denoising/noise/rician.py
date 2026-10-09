"""Rician noise for MRI magnitude images (spec section 7.4).

In MRI, noise is Gaussian in both real and imaginary k-space channels. Taking
the magnitude of the complex image gives a Rician-distributed result:

```text
n_r, n_i  ~  N(0, sigma^2)      (independent)
out        =  sqrt((s + n_r)^2 + n_i^2)
```

At high SNR (s >> sigma) this converges to additive Gaussian; at low SNR the
floor prevents the magnitude from going negative, unlike the Gaussian model.

``sigma`` is the standard deviation of each complex channel in normalised
[0, 1] image units, matching the convention used by the Gaussian generator.
"""

from __future__ import annotations

import numpy as np

from ._common import (
    GrayImage,
    SeedLike,
    resolve_rng,
    to_float01,
    to_uint8,
    validate_image,
    validate_non_negative,
)

__all__ = ["add_rician_noise"]


def add_rician_noise(
    image: GrayImage,
    sigma: float = 0.05,
    seed: SeedLike = None,
) -> GrayImage:
    """Apply Rician noise to a grayscale image (MRI magnitude noise model).

    Draws independent Gaussian noise on the real and imaginary channels, then
    takes the magnitude. ``sigma = 0`` returns the image unchanged.

    Args:
        image: 2-D uint8 grayscale image. Not modified.
        sigma: Standard deviation of each complex Gaussian channel, in
            normalised [0, 1] units, >= 0.
        seed: Int, :class:`numpy.random.Generator`, or ``None`` for fresh
            entropy.

    Returns:
        A new uint8 array with the same shape as *image*.

    Raises:
        TypeError: if *image* is not an array, or *sigma* is not a number.
        ValueError: if *image* is not 2-D uint8, or *sigma* is negative.
    """
    image = validate_image(image)
    sigma = validate_non_negative(sigma, "sigma")
    rng = resolve_rng(seed)

    values = to_float01(image)
    if sigma > 0.0:
        n_r = rng.normal(0.0, sigma, size=values.shape)
        n_i = rng.normal(0.0, sigma, size=values.shape)
        values = np.sqrt((values + n_r) ** 2 + n_i ** 2)
    return to_uint8(values)
