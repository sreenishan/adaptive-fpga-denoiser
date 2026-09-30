#!/usr/bin/env python3
"""CLI: generate the paired (noisy, clean) denoising dataset.

Usage
-----
    python scripts/generate_denoising_dataset.py [--config PATH] [--overwrite] [--quiet]

The script reads source images from ``data/raw/``, applies 12 noise conditions
(4 levels × 3 types) to each, and writes the results to ``data/denoising/``
with a manifest at ``data/denoising/manifest.csv``.

If ``data/raw/`` is empty or absent the script falls back to synthetic test
patterns, which is useful for CI where real photographs are not checked in.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from the repo root without installing the package.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from denoising.config import load_denoising_dataset_config
from denoising.dataset.denoising_generate import generate_denoising_dataset
from denoising.dataset.sources import load_sources, synthetic_sources


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the paired DnCNN denoising dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(_REPO / "configs" / "denoising_dataset.yaml"),
        metavar="PATH",
        help="path to denoising_dataset.yaml (default: configs/denoising_dataset.yaml)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing dataset",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress output",
    )
    p.add_argument(
        "--synthetic-only",
        dest="synthetic_only",
        action="store_true",
        help="use synthetic sources even when data/raw/ contains real images",
    )
    return p.parse_args()


def main() -> int:
    args = _parse()

    config = load_denoising_dataset_config(args.config)
    raw_dir: Path = config.paths.raw_dir

    if args.synthetic_only or not raw_dir.exists() or not any(raw_dir.iterdir()):
        if not args.quiet:
            reason = "--synthetic-only" if args.synthetic_only else f"{raw_dir} is empty or absent"
            print(f"  using synthetic sources ({reason})")
        sources = synthetic_sources(10, config.image, config.split.seed)
    else:
        sources = load_sources(raw_dir, config.image)
        if not args.quiet:
            print(f"  loaded {len(sources)} source images from {raw_dir}")

    t0 = time.monotonic()
    try:
        summary = generate_denoising_dataset(
            sources,
            config,
            overwrite=args.overwrite,
            verbose=not args.quiet,
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("  pass --overwrite to replace the existing dataset", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - t0
    if not args.quiet:
        print(f"\ndone in {elapsed:.1f}s")
        print(f"  sources : {summary.sources}")
        print(f"  samples : {summary.samples}")
        print(f"  splits  : {dict(summary.per_split)}")
        print(f"  by type : {dict(summary.per_noise_type)}")
        print(f"  manifest: {summary.manifest}")
        print(f"  root    : {summary.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
