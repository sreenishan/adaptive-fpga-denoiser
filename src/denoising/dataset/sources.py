"""Clean source images for the dataset (spec section 8).

A source is one clean image. Every noisy sample is derived from exactly one
source, and a source never straddles a split — see
:func:`denoising.dataset.generate.assign_splits`.

Sources come from one of two places:

``data/raw/``
    Real images, the preferred input. Loaded grayscale and resized once, before
    any noise is added.
Synthetic patterns
    Deterministic gradients, sinusoids and shapes, for exercising the pipeline
    when no photographs are available. They are labelled ``synthetic`` in the
    manifest, because a classifier trained on them has been trained on
    synthetic data and any accuracy it reaches is an accuracy on synthetic data.

Images are read and written through ``cv2.imdecode`` / ``cv2.imencode`` rather
than ``imread`` / ``imwrite``: the latter pair route the path through a
non-Unicode C API on Windows and fail on perfectly ordinary directory names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import cv2
import numpy as np

from ..config import ImageConfig
from ..logging_utils import get_logger
from ..noise._common import GrayImage, to_uint8

__all__ = [
    "IMAGE_SUFFIXES",
    "SourceImage",
    "load_sources",
    "synthetic_sources",
    "read_gray_image",
    "write_gray_image",
    "resize_to",
    "check_unique_ids",
]

_LOG = get_logger(__name__)

#: File types accepted from ``data/raw/``.
IMAGE_SUFFIXES: Final[tuple[str, ...]] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)

_SAFE_ID = re.compile(r"[^0-9A-Za-z._-]+")


@dataclass(frozen=True)
class SourceImage:
    """One clean image, already grayscale and at the configured size.

    Attributes:
        source_id: Stable identifier used in filenames and in the manifest.
            Every sample derived from this image carries it, which is what
            makes the leakage check possible.
        origin: Where it came from — a file name, or ``"synthetic"``.
        image: 2-D uint8 array.
    """

    source_id: str
    origin: str
    image: GrayImage


def read_gray_image(path: Path) -> GrayImage | None:
    """Read *path* as an 8-bit grayscale image, or ``None`` if it cannot be
    decoded."""
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        _LOG.warning("could not read %s: %s", path, exc)
        return None
    if raw.size == 0:
        return None
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image.astype(np.uint8, copy=False)


def write_gray_image(path: Path, image: GrayImage) -> None:
    """Write *image* to *path* as a PNG.

    PNG because it is lossless: a JPEG round trip would add compression
    artefacts to the sample and the classifier would learn those too.

    Raises:
        OSError: if encoding fails.
    """
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"failed to encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())


def resize_to(image: GrayImage, width: int, height: int) -> GrayImage:
    """Resize *image* to ``width`` x ``height``.

    ``INTER_AREA`` when shrinking, ``INTER_LINEAR`` when enlarging. Resizing
    happens **before** noise is added: resampling a noisy image low-passes the
    noise and changes its character, which would blur the very distinction the
    classifier is being asked to make.
    """
    if image.shape == (height, width):
        return image
    shrinking = width * height < image.shape[0] * image.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _safe_id(name: str) -> str:
    cleaned = _SAFE_ID.sub("_", name).strip("_")
    return cleaned or "source"


def load_sources(raw_dir: Path, image: ImageConfig) -> list[SourceImage]:
    """Load every image under *raw_dir*, grayscale and resized.

    Files are visited in sorted order so the source list — and therefore the
    split assignment — does not depend on the filesystem's iteration order.
    Undecodable files are reported and skipped rather than aborting a long run.

    Args:
        raw_dir: Directory to search, recursively.
        image: Target geometry from the dataset configuration.

    Returns:
        The loaded sources, possibly empty.
    """
    if not raw_dir.is_dir():
        return []

    paths = sorted(
        (p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.as_posix(),
    )

    sources: list[SourceImage] = []
    used: set[str] = set()
    for path in paths:
        pixels = read_gray_image(path)
        if pixels is None:
            _LOG.warning("skipping %s: not a decodable image", path)
            continue
        base = _safe_id(path.relative_to(raw_dir).with_suffix("").as_posix())
        source_id = base
        suffix = 2
        while source_id in used:
            source_id = f"{base}_{suffix}"
            suffix += 1
        used.add(source_id)
        sources.append(
            SourceImage(
                source_id=source_id,
                origin=path.name,
                image=resize_to(pixels, image.width, image.height),
            )
        )
    return sources


def _synthetic_image(width: int, height: int, rng: np.random.Generator) -> GrayImage:
    """One deterministic pattern: gradient, low-frequency texture, and shapes."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    scale = float(max(width, height))
    xx /= scale
    yy /= scale

    angle = rng.uniform(0.0, 2.0 * np.pi)
    values = np.cos(angle) * xx + np.sin(angle) * yy

    # A few low-frequency sinusoids give structured texture, so the classifier
    # cannot separate the classes on local variance alone.
    for _ in range(int(rng.integers(2, 5))):
        frequency = rng.uniform(1.0, 6.0)
        direction = rng.uniform(0.0, 2.0 * np.pi)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = rng.uniform(0.1, 0.5)
        projection = np.cos(direction) * xx + np.sin(direction) * yy
        values += amplitude * np.sin(2.0 * np.pi * frequency * projection + phase)

    values -= values.min()
    peak = values.max()
    values = values / peak if peak > 0 else np.zeros_like(values)

    # Hard-edged shapes: flat regions and sharp boundaries, which is where
    # median and Gaussian filtering visibly differ.
    for _ in range(int(rng.integers(1, 4))):
        level = rng.uniform(0.0, 1.0)
        if rng.random() < 0.5:
            top = int(rng.integers(0, max(1, height - 1)))
            left = int(rng.integers(0, max(1, width - 1)))
            box_h = int(rng.integers(height // 8 + 1, max(height // 3, height // 8 + 2)))
            box_w = int(rng.integers(width // 8 + 1, max(width // 3, width // 8 + 2)))
            values[top : top + box_h, left : left + box_w] = level
        else:
            centre_y = rng.uniform(0.0, height) / scale
            centre_x = rng.uniform(0.0, width) / scale
            radius = rng.uniform(0.08, 0.25)
            mask = (xx - centre_x) ** 2 + (yy - centre_y) ** 2 <= radius**2
            values[mask] = level

    # Keep away from the rails so the pattern itself is not already clipped.
    low = rng.uniform(0.03, 0.15)
    high = rng.uniform(0.85, 0.97)
    return to_uint8(low + values * (high - low))


def synthetic_sources(
    count: int, image: ImageConfig, seed: int, *, prefix: str = "synthetic"
) -> list[SourceImage]:
    """Build *count* deterministic synthetic source images.

    The same ``seed`` and ``count`` always produce the same images, and image
    *i* depends only on ``(seed, i)``, so appending sources to a dataset does
    not change the ones already generated.

    Args:
        count: How many images to build; must be >= 1.
        image: Target geometry from the dataset configuration.
        seed: Master seed.
        prefix: Leading part of each ``source_id``.

    Returns:
        The generated sources, in index order.

    Raises:
        ValueError: if *count* is less than 1.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    sources: list[SourceImage] = []
    for index in range(count):
        rng = np.random.default_rng([seed, index])
        sources.append(
            SourceImage(
                source_id=f"{prefix}_{index:04d}",
                origin="synthetic",
                image=_synthetic_image(image.width, image.height, rng),
            )
        )
    return sources


def check_unique_ids(sources: Sequence[SourceImage]) -> None:
    """Raise if two sources share an id.

    Two sources with one id would put samples from different images under one
    manifest identity, and the leakage guarantee is stated per id.

    Raises:
        ValueError: if any id appears more than once.
    """
    seen: set[str] = set()
    for source in sources:
        if source.source_id in seen:
            raise ValueError(f"duplicate source_id {source.source_id!r}")
        seen.add(source.source_id)

