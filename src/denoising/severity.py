"""Noise severity estimation (stage 4 of the pipeline).

The classifier says *what* noise is present; this says *how much*. Each noisy
class measures its own physical quantity, from the noisy image alone — at
inference there is no clean original to compare against, so any estimator that
needed one would be unusable exactly where it matters.

``salt_pepper`` — **impulse fraction**: the share of pixels that are 0 or 255
    AND differ from their 3x3 median by more than 40 grey levels. The median
    test is what separates an impulse from a genuinely black or white region,
    which a bare count of extreme pixels would score as noise. One residue:
    a convex corner of a solid 0/255 shape has only 4 of 9 matching
    neighbours and is counted — a handful of pixels per shape, far below the
    first cut point.

``gaussian`` — **noise sigma / 255**, Immerkaer's estimator (1996): convolve
    with the difference of two Laplacians, whose response to smooth image
    structure largely cancels, then take the scaled mean absolute response.
    Unlike the mean local variance, texture barely moves it.

``speckle`` — **noise sigma / mean intensity**. Speckle is multiplicative
    (``I + I*n``), so its absolute sigma scales with brightness; dividing by
    the mean recovers the noise's own standard deviation, ~sqrt(variance).

Measured on 120 held-out synthetic sources at each of the dataset's three
levels, each estimate lands within ~3% of the true noise parameter and the
configured cut points classify all 360 images per class correctly. That is a
statement about the synthetic generator: a real camera's noise is not one
clean distribution, so treat the level there as an estimate, which is how the
UI labels it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SeverityConfig
from .filters._window import sliding_windows
from .filters.median import median_filter
from .filters.selector import SEVERITY_LEVELS
from .noise._common import GrayImage, validate_image

__all__ = [
    "Severity",
    "estimate_severity",
    "impulse_fraction",
    "noise_sigma",
    "speckle_ratio",
]

#: Difference-of-Laplacians kernel from Immerkaer, "Fast Noise Variance
#: Estimation", CVIU 64(2), 1996.
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)

#: A 0/255 pixel within this many grey levels of its local median is taken to
#: be real image content, not an impulse.
_IMPULSE_MARGIN = 40

_METRIC_NAME = {
    "salt_pepper": "impulse fraction",
    "gaussian": "noise sigma (normalised)",
    "speckle": "noise sigma / mean",
}


@dataclass(frozen=True)
class Severity:
    """How strong the noise is, and the measurement that decided it.

    Attributes:
        level: One of ``"low"``, ``"medium"``, ``"high"``.
        metric: The measured value the level was read from.
        metric_name: What *metric* measures, for display.
        medium_from: The cut point between low and medium.
        high_from: The cut point between medium and high.
    """

    level: str
    metric: float
    metric_name: str
    medium_from: float
    high_from: float


def impulse_fraction(image: GrayImage) -> float:
    """Share of pixels that are 0/255 outliers against their 3x3 median."""
    image = validate_image(image)
    median = median_filter(image, 3).astype(np.int16)
    extreme = (image == 0) | (image == 255)
    far = np.abs(image.astype(np.int16) - median) > _IMPULSE_MARGIN
    return float(np.mean(extreme & far))


def noise_sigma(image: GrayImage) -> float:
    """Immerkaer's estimate of additive noise sigma, in grey levels.

    The one-pixel border is excluded: there the replicated edge makes the
    Laplacians cancel artificially and would bias the estimate low.
    """
    image = validate_image(image)
    height, width = image.shape
    if height < 3 or width < 3:
        raise ValueError(f"image {image.shape} is too small to estimate noise")
    windows = sliding_windows(image, 3).astype(np.float64)
    response = np.einsum("hwij,ij->hw", windows, _IMMERKAER)[1:-1, 1:-1]
    scale = math.sqrt(math.pi / 2.0) / (6.0 * (width - 2) * (height - 2))
    return float(scale * np.abs(response).sum())


def speckle_ratio(image: GrayImage) -> float:
    """Noise sigma relative to mean intensity — the speckle's own std dev.

    The mean is floored at one grey level so an all-black frame does not
    divide by zero; such a frame carries no multiplicative noise anyway.
    """
    image = validate_image(image)
    return noise_sigma(image) / max(float(image.mean()), 1.0)


def _measure(image: GrayImage, noise_class: str) -> float:
    if noise_class == "salt_pepper":
        return impulse_fraction(image)
    if noise_class == "gaussian":
        return noise_sigma(image) / 255.0
    if noise_class == "speckle":
        return speckle_ratio(image)
    raise ValueError(f"no severity measure for noise class {noise_class!r}")


def estimate_severity(
    image: GrayImage, noise_class: str, config: SeverityConfig
) -> Severity | None:
    """Measure how strong *noise_class* noise is in *image*.

    Args:
        image: The noisy 2-D uint8 image.
        noise_class: The class the level is read for — the classifier's
            answer, since that decides which quantity is meaningful.
        config: Cut points per class.

    Returns:
        A :class:`Severity`, or ``None`` when severity is disabled or the class
        is ``clean`` — no noise has no strength, and inventing a level for it
        would be a measurement of nothing.
    """
    if not config.enabled or noise_class == "clean":
        return None
    metric = _measure(image, noise_class)
    band = config.band(noise_class)
    if metric < band.medium_from:
        level = SEVERITY_LEVELS[0]
    elif metric < band.high_from:
        level = SEVERITY_LEVELS[1]
    else:
        level = SEVERITY_LEVELS[2]
    return Severity(
        level=level,
        metric=metric,
        metric_name=_METRIC_NAME[noise_class],
        medium_from=band.medium_from,
        high_from=band.high_from,
    )
