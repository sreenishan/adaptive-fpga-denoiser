"""Paired (noisy, clean) dataset generation for the DnCNN denoiser.

Separate from the classifier dataset (``generate.py``).  Each source image
produces one clean PNG and 12 noisy PNGs — 4 levels × 3 noise types.

Manifest columns
----------------
``noisy_path``    path to the noisy image (relative to repo root)
``clean_path``    path to the clean image (same source_id)
``split``         train / val / test
``source_id``     stable identifier shared by all images from one source
``noise_type``    salt_pepper | gaussian | speckle
``noise_level``   percentage label: 5 | 10 | 15 | 20
``noise_parameter``  the float actually passed to the generator
``seed``          the RNG seed used

Design rules
------------
- Split is assigned **per source** — all 12 noisy variants of one source land
  in the same split, so no near-duplicate leaks into the test set.
- Seeds are derived by hashing (master_seed, source_id, noise_type, level_index)
  exactly as in ``generate.py``, so they are stable across re-runs.
- Clean images are **not duplicated** in the manifest.  The manifest row for
  each noisy image carries both ``noisy_path`` and ``clean_path``, so the
  loader can find both with a single row read.
"""

from __future__ import annotations

import csv
import hashlib

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import numpy as np

from ..config import (
    DENOISING_NOISE_TYPES,
    PROJECT_ROOT,
    DenoisingDatasetConfig,
)
from ..logging_utils import get_logger
from ..noise import add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise
from ..noise._common import GrayImage
from .sources import SourceImage, check_unique_ids, write_gray_image

__all__ = [
    "DENOISING_MANIFEST_COLUMNS",
    "DenoisingPlannedSample",
    "DenoisingGenerationSummary",
    "denoising_sample_seed",
    "plan_denoising_dataset",
    "render_denoising_sample",
    "generate_denoising_dataset",
    "write_denoising_manifest",
]

_LOG = get_logger(__name__)

DENOISING_MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "noisy_path",
    "clean_path",
    "split",
    "source_id",
    "noise_type",
    "noise_level",
    "noise_parameter",
    "seed",
)

_SEED_MODULUS: Final[int] = 2**32
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")


@dataclass(frozen=True)
class DenoisingPlannedSample:
    """One (noisy, clean) pair, decided but not yet rendered."""

    source_id: str
    split: str
    noise_type: str
    noise_level: int              # percentage label: 5 / 10 / 15 / 20
    level_index: int              # index into the levels list (0-3)
    noise_parameter: float        # the primary float value (amount/sigma/variance)
    noise_parameters: Mapping[str, float]
    seed: int
    noisy_relative_path: str
    clean_relative_path: str


@dataclass(frozen=True)
class DenoisingGenerationSummary:
    samples: int
    sources: int
    per_split: Mapping[str, int]
    per_noise_type: Mapping[str, int]
    manifest: Path
    root: Path


def denoising_sample_seed(
    master_seed: int, source_id: str, noise_type: str, level_index: int
) -> int:
    """Stable seed for one (source, noise_type, level) triple."""
    material = f"dn|{master_seed}|{source_id}|{noise_type}|{level_index}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=4).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


def _assign_splits(
    source_ids: Sequence[str], config: DenoisingDatasetConfig
) -> dict[str, str]:
    unique = sorted(dict.fromkeys(source_ids))
    if len(unique) != len(source_ids):
        raise ValueError("source_ids contains duplicates")
    if len(unique) < 3:
        raise ValueError("need at least 3 sources to fill train/val/test")

    ratios = (
        config.split.train_ratio,
        config.split.validation_ratio,
        config.split.test_ratio,
    )
    total = len(unique)
    exact = [r * total for r in ratios]
    counts = [max(1, int(v)) for v in exact]

    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    idx = 0
    while remainder > 0:
        counts[order[idx % 3]] += 1
        remainder -= 1
        idx += 1
    while remainder < 0:
        largest = max(range(3), key=lambda i: counts[i])
        if counts[largest] <= 1:
            break
        counts[largest] -= 1
        remainder += 1

    rng = np.random.default_rng(config.split.seed)
    shuffled = [unique[i] for i in rng.permutation(len(unique))]
    assignment: dict[str, str] = {}
    pos = 0
    for split, count in zip(SPLITS, counts):
        for sid in shuffled[pos: pos + count]:
            assignment[sid] = split
        pos += count
    return assignment


def plan_denoising_dataset(
    source_ids: Sequence[str], config: DenoisingDatasetConfig
) -> list[DenoisingPlannedSample]:
    """Plan every (noisy, clean) pair without touching pixels."""
    assignment = _assign_splits(source_ids, config)
    noise_cfg = config.noise
    master_seed = config.split.seed
    planned: list[DenoisingPlannedSample] = []

    for source_id in source_ids:
        split = assignment[source_id]
        clean_rel = f"{split}/clean/{source_id}__clean.png"

        for noise_type in DENOISING_NOISE_TYPES:
            for level_idx, level_pct in enumerate(noise_cfg.levels):
                params = noise_cfg.parameters_for(noise_type, level_idx)
                primary = noise_cfg.primary_parameter(noise_type, level_idx)
                seed = denoising_sample_seed(master_seed, source_id, noise_type, level_idx)
                noisy_rel = (
                    f"{split}/noisy/"
                    f"{source_id}__{noise_type}__{level_pct}.png"
                )
                planned.append(
                    DenoisingPlannedSample(
                        source_id=source_id,
                        split=split,
                        noise_type=noise_type,
                        noise_level=level_pct,
                        level_index=level_idx,
                        noise_parameter=primary,
                        noise_parameters=dict(params),
                        seed=seed,
                        noisy_relative_path=noisy_rel,
                        clean_relative_path=clean_rel,
                    )
                )
    return planned


def render_denoising_sample(sample: DenoisingPlannedSample, clean: GrayImage) -> GrayImage:
    """Apply the noise model described by *sample* to *clean*."""
    params = dict(sample.noise_parameters)
    if sample.noise_type == "salt_pepper":
        return add_salt_pepper_noise(
            clean, params["amount"], params["salt_vs_pepper"], sample.seed
        )
    if sample.noise_type == "gaussian":
        return add_gaussian_noise(clean, params["mean"], params["sigma"], sample.seed)
    if sample.noise_type == "speckle":
        return add_speckle_noise(clean, params["variance"], sample.seed)
    raise ValueError(f"no generator for noise type {sample.noise_type!r}")


def _manifest_path(root: Path, relative_path: str) -> str:
    absolute = root / relative_path
    try:
        return absolute.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return relative_path


def write_denoising_manifest(
    path: Path, samples: Iterable[DenoisingPlannedSample], root: Path
) -> int:
    """Write the manifest CSV; return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(DENOISING_MANIFEST_COLUMNS)
        for s in samples:
            writer.writerow([
                _manifest_path(root, s.noisy_relative_path),
                _manifest_path(root, s.clean_relative_path),
                s.split,
                s.source_id,
                s.noise_type,
                s.noise_level,
                s.noise_parameter,
                s.seed,
            ])
            rows += 1
    return rows


def _existing_outputs(root: Path) -> list[Path]:
    found: list[Path] = []
    for split in SPLITS:
        for subdir in ("clean", "noisy"):
            d = root / split / subdir
            if d.is_dir():
                found.extend(sorted(p for p in d.iterdir() if p.suffix == ".png"))
    return found


def generate_denoising_dataset(
    sources: Sequence[SourceImage],
    config: DenoisingDatasetConfig,
    *,
    root: Path | None = None,
    manifest: Path | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> DenoisingGenerationSummary:
    """Generate the full paired denoising dataset.

    Args:
        sources:   Clean source images already grayscale and resized.
        config:    Loaded denoising dataset config.
        root:      Dataset root directory.
        manifest:  Manifest CSV path.
        overwrite: Replace an existing dataset.
        verbose:   Print progress.

    Returns:
        :class:`DenoisingGenerationSummary`
    """
    if not sources:
        raise ValueError("no source images")
    check_unique_ids(sources)

    root = (root or config.paths.dataset_dir).resolve()
    manifest_path = (manifest or config.paths.manifest).resolve()

    existing = _existing_outputs(root)
    if existing and not overwrite:
        raise FileExistsError(
            f"{root} already holds {len(existing)} generated files; "
            "pass overwrite=True to replace them"
        )
    if existing:
        _LOG.info("removing %d previously generated files under %s", len(existing), root)
        for p in existing:
            p.unlink()

    by_id = {s.source_id: s for s in sources}
    source_ids = [s.source_id for s in sources]
    planned = plan_denoising_dataset(source_ids, config)

    # Write clean images (one per source per split)
    clean_written: set[str] = set()
    for sample in planned:
        if sample.source_id not in clean_written:
            clean_path = root / sample.clean_relative_path
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            write_gray_image(clean_path, by_id[sample.source_id].image)
            clean_written.add(sample.source_id)

    # Write noisy images
    for i, sample in enumerate(planned):
        noisy_path = root / sample.noisy_relative_path
        noisy_path.parent.mkdir(parents=True, exist_ok=True)
        noisy_img = render_denoising_sample(sample, by_id[sample.source_id].image)
        write_gray_image(noisy_path, noisy_img)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(planned)} noisy images written")

    rows = write_denoising_manifest(manifest_path, planned, root)

    per_split: dict[str, int] = {s: 0 for s in SPLITS}
    per_type: dict[str, int] = {t: 0 for t in DENOISING_NOISE_TYPES}
    for s in planned:
        per_split[s.split] += 1
        per_type[s.noise_type] += 1

    summary = DenoisingGenerationSummary(
        samples=rows,
        sources=len(sources),
        per_split=per_split,
        per_noise_type=per_type,
        manifest=manifest_path,
        root=root,
    )
    _LOG.info(
        "generated %d paired samples from %d sources", summary.samples, summary.sources
    )
    return summary
