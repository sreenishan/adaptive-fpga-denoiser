#!/usr/bin/env python3
"""CLI: train the DnCNN denoiser.

Usage
-----
    python scripts/train_denoiser.py [--manifest PATH] [--config PATH] [--quiet]

Reads the paired denoising dataset (must be generated first) and trains a
DnCNN network.  The best checkpoint (by validation PSNR) is saved to
``models/checkpoints/denoiser.pt``.

Generate the dataset first::

    python scripts/generate_denoising_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from denoising.config import load_denoising_training_config
from denoising.model.train_denoiser import train_denoiser


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the DnCNN denoiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        default=str(_REPO / "data" / "denoising" / "manifest.csv"),
        metavar="PATH",
        help="path to the denoising manifest CSV (default: data/denoising/manifest.csv)",
    )
    p.add_argument(
        "--config",
        default=str(_REPO / "configs" / "denoising_training.yaml"),
        metavar="PATH",
        help="path to denoising_training.yaml (default: configs/denoising_training.yaml)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-epoch progress",
    )
    return p.parse_args()


def main() -> int:
    args = _parse()
    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"error: manifest not found at {manifest}", file=sys.stderr)
        print("  run: python scripts/generate_denoising_dataset.py", file=sys.stderr)
        return 1

    config = load_denoising_training_config(args.config)
    if not args.quiet:
        print(f"training DnCNN depth={config.model.depth} filters={config.model.filters}")
        print(f"  manifest : {manifest}")
        print(f"  epochs   : {config.epochs}")
        print(f"  device   : {config.device}")
        print(f"  checkpoint will be saved to: {config.checkpoint_path}")
        print()

    train_denoiser(manifest, config, verbose=not args.quiet)

    if not args.quiet:
        print(f"\ncheckpoint saved: {config.checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
