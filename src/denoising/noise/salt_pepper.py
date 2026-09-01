"""Salt-and-pepper (impulse) noise (spec section 7.1).

A fraction of the pixels are replaced outright with the extreme values 255
(salt) or 0 (pepper). The rest are untouched — this is impulse noise, not an
additive perturbation, which is why the median filter removes it so cleanly and
why an averaging filter smears it instead.
"""

from __future__ import annotations

from ._common import (
    PIXEL_MAX,
    GrayImage,
    SeedLike,
    resolve_rng,
    validate_image,
    validate_unit_interval,
)

__all__ = ["add_salt_pepper_noise"]


def add_salt_pepper_noise(
    image: GrayImage,
    amount: float,
    salt_vs_pepper: float = 0.5,
    seed: SeedLike = None,
) -> GrayImage:
    """Replace a fraction of the pixels with 0 or 255.

    Exactly ``round(amount * image.size)`` distinct pixels are corrupted, chosen
    without replacement. A per-pixel coin flip would give that count only on
    average; a fixed count means the label "5% salt and pepper" describes every
    sample rather than the distribution they were drawn from.

    Args:
        image: 2-D uint8 grayscale image. Not modified.
        amount: Fraction of pixels to corrupt, in [0, 1].
        salt_vs_pepper: Fraction of the corrupted pixels set to 255 rather than
            0, in [0, 1]. 0.5 gives equal salt and pepper; 1.0 gives salt only.
        seed: Int, :class:`numpy.random.Generator`, or ``None`` for fresh
            entropy. An int or a Generator makes the result reproducible.

    Returns:
        A new uint8 array with the same shape as *image*.

    Raises:
        TypeError: if *image* is not an array, or a parameter is not a number.
        ValueError: if *image* is not 2-D uint8, or a parameter is out of range.
    """
    image = validate_image(image)
    amount = validate_unit_interval(amount, "amount")
    salt_vs_pepper = validate_unit_interval(salt_vs_pepper, "salt_vs_pepper")
    rng = resolve_rng(seed)

    out = image.copy()
    total = image.size
    corrupted = int(round(amount * total))
    if corrupted == 0:
        return out

    flat_indices = rng.choice(total, size=corrupted, replace=False)
    salt_count = int(round(corrupted * salt_vs_pepper))

    flat = out.reshape(-1)
    flat[flat_indices[:salt_count]] = PIXEL_MAX
    flat[flat_indices[salt_count:]] = 0
    return out

