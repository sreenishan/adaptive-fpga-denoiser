"""Shared helpers for the noise generators.

Private to :mod:`denoising.noise`. It exists so the three generators agree on
what a valid image is, how a seed becomes a random generator, and how a float
result is rounded back to uint8 — three copies of that logic would drift, and
the drift would show up as an unreproducible dataset.

Intensity convention
--------------------
``sigma`` and ``variance`` are expressed in **normalised [0, 1] image units**,
matching ``configs/dataset.yaml``. A sigma of 0.08 is 8% of full scale, i.e.
about 20 grey levels — not 0.08 of a grey level.
"""

from __future__ import annotations

from typing import Final, Union

import numpy as np
import numpy.typing as npt

__all__ = [
    "PIXEL_MAX",
    "GrayImage",
    "SeedLike",
    "resolve_rng",
    "validate_image",
    "validate_unit_interval",
    "validate_non_negative",
    "validate_finite",
    "to_float01",
    "to_uint8",
]

#: Largest value an 8-bit pixel can hold.
PIXEL_MAX: Final[int] = int(np.iinfo(np.uint8).max)

#: A grayscale image: 2-D, uint8, 0-255.
GrayImage = npt.NDArray[np.uint8]

#: Anything accepted as a source of randomness. ``None`` means fresh entropy,
#: so the result is *not* reproducible; an int or a Generator makes it so.
SeedLike = Union[int, np.random.Generator, None]


def resolve_rng(seed: SeedLike) -> np.random.Generator:
    """Return a NumPy generator for *seed*.

    A :class:`numpy.random.Generator` is passed through unchanged, so a caller
    generating a whole dataset can thread one stream through many calls instead
    of reseeding — reseeding from a counter correlates the streams.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def validate_image(image: object, name: str = "image") -> GrayImage:
    """Check that *image* is a non-empty 2-D uint8 array and return it.

    Raises:
        TypeError: if *image* is not a NumPy array.
        ValueError: if it is not 2-D uint8, or is empty.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {image.dtype}")
    if image.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D grayscale, got {image.ndim} dimensions "
            "(colour is out of scope for the initial system)"
        )
    if image.size == 0:
        raise ValueError(f"{name} must not be empty")
    return image


def validate_unit_interval(value: float, name: str) -> float:
    """Check that *value* lies in [0, 1] and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def validate_non_negative(value: float, name: str) -> float:
    """Check that *value* is finite and >= 0, and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def validate_finite(value: float, name: str) -> float:
    """Check that *value* is a finite number and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def to_float01(image: GrayImage) -> npt.NDArray[np.float64]:
    """Convert a uint8 image to float64 in [0, 1]."""
    return image.astype(np.float64) / PIXEL_MAX


def to_uint8(values: npt.NDArray[np.float64]) -> GrayImage:
    """Clip float values in [0, 1] back to uint8.

    Clipping happens **before** scaling, so nothing can wrap around: adding
    noise to a 255 pixel saturates at 255 rather than becoming 3. Rounding is
    :func:`numpy.rint` (half to even), which is the single rounding rule used
    everywhere a float becomes a pixel in this project.
    """
    return np.rint(np.clip(values, 0.0, 1.0) * PIXEL_MAX).astype(np.uint8)
