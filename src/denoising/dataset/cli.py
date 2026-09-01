"""Command line entry point for dataset generation.

``python scripts/generate_dataset.py`` builds the four-class dataset from the
images in ``data/raw/``, or from deterministic synthetic patterns when asked
explicitly.

Two guards are deliberate. Synthetic sources are never a silent fallback: an
empty ``data/raw/`` is an error naming both options, because a dataset built
from patterns nobody chose to build would be reported as an ordinary result. And
an existing dataset is never merged into — ``--overwrite`` replaces it, and
without that flag a populated output directory stops the run, since a half-old
dataset with one manifest describing all of it is worse than either half.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from ..config import CONFIG_DIR, ConfigError, DatasetConfig, load_dataset_config
from ..logging_utils import configure_logging, get_logger
from .generate import SPLITS, class_counts, generate_dataset, plan_dataset
from .sources import SourceImage, load_sources, synthetic_sources

__all__ = ["main"]

_LOG = get_logger(__name__)


def _collect_sources(
    dataset: DatasetConfig, synthetic: int | None, raw_dir: Path | None
) -> list[SourceImage]:
    """Load raw images, or build synthetic ones when explicitly requested."""
    if synthetic is not None:
        _LOG.info("building %d synthetic source images", synthetic)
        return synthetic_sources(synthetic, dataset.image, dataset.split.seed)

    directory = raw_dir or dataset.paths.raw_dir
    sources = load_sources(directory, dataset.image)
    if not sources:
        raise FileNotFoundError(
            f"no images found in {directory}. Put clean images there, or pass "
            "--synthetic N to generate N synthetic sources instead (they are "
            "labelled synthetic in the manifest)."
        )
    _LOG.info("loaded %d source images from %s", len(sources), directory)
    return sources


def _print_plan(dataset: DatasetConfig, sources: Sequence[SourceImage]) -> None:
    """Print what would be generated, without writing anything."""
    planned = plan_dataset([source.source_id for source in sources], dataset)
    per_split: dict[str, dict[str, int]] = {}
    for sample in planned:
        per_split.setdefault(sample.split, {})[sample.noise_type] = (
            per_split.setdefault(sample.split, {}).get(sample.noise_type, 0) + 1
        )

    print(f"sources        : {len(sources)}")
    print(f"planned samples: {len(planned)}")
    print()
    print(f"{'split':<8}{'clean':>8}{'salt_pepper':>14}{'gaussian':>10}{'speckle':>9}{'total':>8}")
    for split in SPLITS:
        counts = per_split.get(split, {})
        row = [counts.get(name, 0) for name in ("clean", "salt_pepper", "gaussian", "speckle")]
        print(f"{split:<8}{row[0]:>8}{row[1]:>14}{row[2]:>10}{row[3]:>9}{sum(row):>8}")
    totals = class_counts(planned)
    print()
    print("class totals   : " + ", ".join(f"{k}={v}" for k, v in totals.items()))
    print(
        "note: the clean class has one sample per source while each noisy class "
        "has one per configured intensity, so the classes are imbalanced by "
        "design. Training must weight them."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the dataset. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="denoising-generate-dataset",
        description="Generate the four-class noise dataset and its manifest.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "dataset.yaml",
        help="dataset configuration file (default: configs/dataset.yaml)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="directory of clean source images (default: the configured raw_dir)",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=None,
        metavar="N",
        help="generate N synthetic source images instead of reading raw images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="dataset root (default: the configured generated_dir)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest CSV path (default: the configured manifest)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing generated dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and write nothing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    configure_logging(getattr(logging, args.log_level))

    try:
        dataset = load_dataset_config(args.config)
        sources = _collect_sources(dataset, args.synthetic, args.raw_dir)
        if args.dry_run:
            _print_plan(dataset, sources)
            return 0
        summary = generate_dataset(
            sources,
            dataset,
            root=args.output,
            manifest=args.manifest,
            overwrite=args.overwrite,
        )
    except (ConfigError, FileNotFoundError, FileExistsError, ValueError) as exc:
        _LOG.error("%s", exc)
        return 1

    print(f"dataset root   : {summary.root}")
    print(f"manifest       : {summary.manifest}")
    print(f"sources        : {summary.sources}")
    print(f"samples        : {summary.samples}")
    print("per split      : " + ", ".join(f"{k}={v}" for k, v in summary.per_split.items()))
    print("per class      : " + ", ".join(f"{k}={v}" for k, v in summary.per_class.items()))
    return 0
