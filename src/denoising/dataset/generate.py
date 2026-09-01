"""Dataset generation and splitting (spec sections 8 and 9).

One clean source image produces one ``clean`` sample plus one sample for every
configured intensity of every noise model. The plan is computed first and the
pixels are produced second, so the split, the seeds and the file layout can be
inspected — and tested — without writing a single image.

Two properties this module exists to guarantee:

**No leakage.** The split is assigned per *source*, never per sample. Every
noisy version of one clean image lands in the same split; otherwise the test set
contains near-copies of training images and the reported accuracy measures
memorisation.

**Reproducible seeds.** A sample's seed is derived by hashing
``(master_seed, source_id, noise_type, level)`` rather than drawn from a running
counter, so it does not change when sources are added, removed or reordered, and
one sample can be regenerated on its own.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import numpy as np

from ..config import CLASSES, PROJECT_ROOT, DatasetConfig
from ..logging_utils import get_logger
from ..noise import add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise
from ..noise._common import GrayImage
from .sources import SourceImage, check_unique_ids, write_gray_image

__all__ = [
    "MANIFEST_COLUMNS",
    "SPLITS",
    "PlannedSample",
    "GenerationSummary",
    "sample_seed",
    "assign_splits",
    "plan_dataset",
    "render_sample",
    "generate_dataset",
    "write_manifest",
    "existing_outputs",
    "class_counts",
]

_LOG = get_logger(__name__)

#: Manifest columns, in order, fixed by spec section 9.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "path",
    "split",
    "label",
    "source_id",
    "noise_type",
    "noise_parameter",
    "seed",
)

#: Split names, in the order sources are handed out.
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")

_SEED_MODULUS: Final[int] = 2**32


@dataclass(frozen=True)
class PlannedSample:
    """One sample, decided but not yet rendered.

    Attributes:
        source_id: The clean image this is derived from.
        split: ``train``, ``val`` or ``test``.
        noise_type: One of :data:`denoising.config.CLASSES`.
        label: Index of *noise_type* in ``CLASSES`` — the integer label.
        noise_parameter: The primary intensity (``amount``, ``sigma`` or
            ``variance``), or ``None`` for the clean class. The full parameter
            set is in :attr:`noise_parameters`.
        noise_parameters: Every argument passed to the generator.
        seed: Seed handed to the generator.
        relative_path: Path of the image relative to the dataset root, in POSIX
            form so the manifest survives being read on another platform.
    """

    source_id: str
    split: str
    noise_type: str
    label: int
    noise_parameter: float | None
    noise_parameters: Mapping[str, float]
    seed: int
    relative_path: str


@dataclass(frozen=True)
class GenerationSummary:
    """What a generation run produced."""

    samples: int
    sources: int
    per_split: Mapping[str, int]
    per_class: Mapping[str, int]
    manifest: Path
    root: Path


def sample_seed(master_seed: int, source_id: str, noise_type: str, level: int) -> int:
    """Derive a stable per-sample seed.

    BLAKE2b of the four inputs, not a counter: adding a source must not change
    the noise on every sample that follows it. Stable across processes and
    Python versions, unlike :func:`hash`.

    Args:
        master_seed: ``dataset.split.seed`` from the configuration.
        source_id: The source image identifier.
        noise_type: One of :data:`denoising.config.CLASSES`.
        level: Index into the configured intensity list; 0 for ``clean``.

    Returns:
        An integer in ``[0, 2**32)``.
    """
    material = f"{master_seed}|{source_id}|{noise_type}|{level}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=4).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


def assign_splits(source_ids: Sequence[str], dataset: DatasetConfig) -> dict[str, str]:
    """Assign each source to exactly one split.

    Sources are sorted, shuffled with the configured seed and handed out by
    largest remainder, so the counts sum to the number of sources exactly and
    each split gets at least one source. The result depends only on the set of
    ids and the seed, not on the order they arrive in.

    Adding or removing a source reshuffles the assignment — the permutation of
    n items is not stable under insertion. That is why a dataset is regenerated
    whole rather than extended.

    Args:
        source_ids: Every source identifier, in any order.
        dataset: Dataset configuration supplying the ratios and the seed.

    Returns:
        Mapping of ``source_id`` to split name.

    Raises:
        ValueError: if there are fewer than three sources — with two, one of
            the three splits would be empty and an empty test set reports no
            error at all, it just reports nothing.
    """
    # Sorted, so the assignment depends on the SET of sources and the seed,
    # never on the order they were handed in — a renamed directory or a
    # different filesystem must not silently reshuffle the splits.
    unique = sorted(dict.fromkeys(source_ids))
    if len(unique) != len(source_ids):
        raise ValueError("source_ids contains duplicates")
    if len(unique) < len(SPLITS):
        raise ValueError(
            f"need at least {len(SPLITS)} sources to fill {SPLITS}, got {len(unique)}"
        )

    ratios = (
        dataset.split.train_ratio,
        dataset.split.validation_ratio,
        dataset.split.test_ratio,
    )
    total = len(unique)
    exact = [ratio * total for ratio in ratios]
    counts = [max(1, int(value)) for value in exact]

    # Hand out what rounding left over, largest fractional part first.
    remainder = total - sum(counts)
    order = sorted(range(len(SPLITS)), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    index = 0
    while remainder > 0:
        counts[order[index % len(order)]] += 1
        remainder -= 1
        index += 1
    while remainder < 0:
        # Only reachable when the minimum of one per split over-allocates.
        largest = max(range(len(SPLITS)), key=lambda i: counts[i])
        if counts[largest] <= 1:
            break
        counts[largest] -= 1
        remainder += 1

    rng = np.random.default_rng(dataset.split.seed)
    shuffled = [unique[index] for index in rng.permutation(len(unique))]

    assignment: dict[str, str] = {}
    position = 0
    for split, count in zip(SPLITS, counts):
        for source_id in shuffled[position : position + count]:
            assignment[source_id] = split
        position += count
    return assignment


def _noise_levels(dataset: DatasetConfig) -> dict[str, list[Mapping[str, float]]]:
    """The configured parameter sets for each noisy class, in level order."""
    noise = dataset.noise
    return {
        "salt_pepper": [
            {"amount": amount, "salt_vs_pepper": noise.salt_pepper.salt_vs_pepper}
            for amount in noise.salt_pepper.amounts
        ],
        "gaussian": [
            {"mean": noise.gaussian.mean, "sigma": sigma} for sigma in noise.gaussian.sigmas
        ],
        "speckle": [{"variance": variance} for variance in noise.speckle.variances],
    }


_PRIMARY_PARAMETER: Final[Mapping[str, str]] = {
    "salt_pepper": "amount",
    "gaussian": "sigma",
    "speckle": "variance",
}


def plan_dataset(
    source_ids: Sequence[str], dataset: DatasetConfig
) -> list[PlannedSample]:
    """Decide every sample: its split, class, parameters, seed and path.

    No pixels are touched. The plan is deterministic given the source ids and
    the configuration.

    Args:
        source_ids: Every source identifier.
        dataset: Dataset configuration.

    Returns:
        The planned samples, ordered by source and then by class.
    """
    assignment = assign_splits(source_ids, dataset)
    levels = _noise_levels(dataset)
    master_seed = dataset.split.seed

    planned: list[PlannedSample] = []
    for source_id in source_ids:
        split = assignment[source_id]
        planned.append(
            PlannedSample(
                source_id=source_id,
                split=split,
                noise_type="clean",
                label=CLASSES.index("clean"),
                noise_parameter=None,
                noise_parameters={},
                seed=sample_seed(master_seed, source_id, "clean", 0),
                relative_path=f"{split}/clean/{source_id}__clean.png",
            )
        )
        for noise_type in CLASSES[1:]:
            for level, parameters in enumerate(levels[noise_type]):
                planned.append(
                    PlannedSample(
                        source_id=source_id,
                        split=split,
                        noise_type=noise_type,
                        label=CLASSES.index(noise_type),
                        noise_parameter=parameters[_PRIMARY_PARAMETER[noise_type]],
                        noise_parameters=dict(parameters),
                        seed=sample_seed(master_seed, source_id, noise_type, level),
                        relative_path=(
                            f"{split}/{noise_type}/{source_id}__{noise_type}__L{level}.png"
                        ),
                    )
                )
    return planned


def render_sample(sample: PlannedSample, clean: GrayImage) -> GrayImage:
    """Apply *sample*'s noise model to *clean*.

    Args:
        sample: The planned sample.
        clean: The source image, already grayscale and at the target size.

    Returns:
        A new uint8 image. For the clean class this is a copy of *clean*.

    Raises:
        ValueError: if the sample names a class with no generator.
    """
    parameters = dict(sample.noise_parameters)
    if sample.noise_type == "clean":
        return clean.copy()
    if sample.noise_type == "salt_pepper":
        return add_salt_pepper_noise(
            clean, parameters["amount"], parameters["salt_vs_pepper"], sample.seed
        )
    if sample.noise_type == "gaussian":
        return add_gaussian_noise(clean, parameters["mean"], parameters["sigma"], sample.seed)
    if sample.noise_type == "speckle":
        return add_speckle_noise(clean, parameters["variance"], sample.seed)
    raise ValueError(f"no generator for noise type {sample.noise_type!r}")


def _sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def _write_sidecar(path: Path, sample: PlannedSample, origin: str) -> None:
    """Write the per-sample metadata required by spec section 8.

    Written from the same record that produces the manifest row, in the same
    pass, so the two cannot describe different things.
    """
    payload = {
        "source": origin,
        "source_id": sample.source_id,
        "class": sample.noise_type,
        "label": sample.label,
        "split": sample.split,
        "noise_parameters": dict(sample.noise_parameters),
        "seed": sample.seed,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(path: Path, samples: Iterable[PlannedSample], root: Path) -> int:
    """Write the manifest CSV and return the number of rows.

    ``path`` column entries are relative to the repository root when the
    dataset lives inside it, and to the dataset root otherwise — either way
    they are relative, so a checkout that moves still resolves.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for sample in samples:
            writer.writerow(
                [
                    _manifest_path(root, sample.relative_path),
                    sample.split,
                    sample.label,
                    sample.source_id,
                    sample.noise_type,
                    "" if sample.noise_parameter is None else sample.noise_parameter,
                    sample.seed,
                ]
            )
            rows += 1
    return rows


def _manifest_path(root: Path, relative_path: str) -> str:
    """The manifest's ``path`` value for one sample."""
    absolute = root / relative_path
    try:
        return absolute.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return relative_path


def class_counts(samples: Iterable[PlannedSample]) -> dict[str, int]:
    """Count samples per class, in ``CLASSES`` order."""
    counts = {name: 0 for name in CLASSES}
    for sample in samples:
        counts[sample.noise_type] += 1
    return counts


def _split_counts(samples: Iterable[PlannedSample]) -> dict[str, int]:
    counts = {name: 0 for name in SPLITS}
    for sample in samples:
        counts[sample.split] += 1
    return counts


def existing_outputs(root: Path) -> list[Path]:
    """Generated files already present under *root*.

    Only the files this module writes are listed — images and sidecars inside
    ``<split>/<class>/`` — so a stray note left in the directory is neither
    reported nor, later, deleted.
    """
    found: list[Path] = []
    for split in SPLITS:
        for name in CLASSES:
            directory = root / split / name
            if not directory.is_dir():
                continue
            found.extend(sorted(p for p in directory.iterdir() if p.suffix in (".png", ".json")))
    return found


def generate_dataset(
    sources: Sequence[SourceImage],
    dataset: DatasetConfig,
    *,
    root: Path | None = None,
    manifest: Path | None = None,
    overwrite: bool = False,
) -> GenerationSummary:
    """Generate the whole dataset: images, sidecars and manifest.

    Args:
        sources: Clean source images, already grayscale and resized.
        dataset: Dataset configuration.
        root: Dataset root; defaults to ``dataset.paths.generated_dir``.
        manifest: Manifest path; defaults to ``dataset.paths.manifest``.
        overwrite: Replace an existing generated dataset. Without it, a
            non-empty output directory is an error rather than a merge — a
            half-old, half-new dataset with one manifest describing it is worse
            than either.

    Returns:
        A :class:`GenerationSummary`.

    Raises:
        ValueError: if *sources* is empty, contains duplicate ids, or there are
            too few sources to fill the splits.
        FileExistsError: if output already exists and *overwrite* is false.
    """
    if not sources:
        raise ValueError("no source images: nothing to generate")
    check_unique_ids(sources)

    root = (root or dataset.paths.generated_dir).resolve()
    manifest = (manifest or dataset.paths.manifest).resolve()

    existing = existing_outputs(root)
    if existing and not overwrite:
        raise FileExistsError(
            f"{root} already holds {len(existing)} generated files; "
            "pass overwrite=True to replace them"
        )
    if existing:
        _LOG.info("removing %d previously generated files under %s", len(existing), root)
        for path in existing:
            path.unlink()

    by_id = {source.source_id: source for source in sources}
    planned = plan_dataset([source.source_id for source in sources], dataset)

    for sample in planned:
        image_path = root / sample.relative_path
        source = by_id[sample.source_id]
        write_gray_image(image_path, render_sample(sample, source.image))
        _write_sidecar(_sidecar_path(image_path), sample, source.origin)

    rows = write_manifest(manifest, planned, root)
    summary = GenerationSummary(
        samples=rows,
        sources=len(sources),
        per_split=_split_counts(planned),
        per_class=class_counts(planned),
        manifest=manifest,
        root=root,
    )
    _LOG.info(
        "generated %d samples from %d sources: %s",
        summary.samples,
        summary.sources,
        ", ".join(f"{k}={v}" for k, v in summary.per_split.items()),
    )
    return summary
