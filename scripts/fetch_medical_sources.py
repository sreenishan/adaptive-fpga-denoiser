"""Download medical source images into data/raw/{ultrasound,mri,ct}/.

Uses MedMNIST (https://medmnist.com) — a free, no-registration collection of
standardised 2D medical images.  Install the package first:

    pip install medmnist

Three modalities are fetched:

    data/raw/ultrasound/   BreastMNIST  — breast ultrasound (grayscale)
    data/raw/mri/          OCTMNIST     — retinal OCT (treated as MRI surrogate)
    data/raw/ct/           ChestMNIST   — chest X-ray (grayscale)

Images are saved as PNG; existing files are skipped so the script is safe to
re-run.  The dataset generator (``scripts/generate_dataset.py``) reads these
images and applies synthetic noise; subdirectory names are picked up by
``load_sources`` in ``src/denoising/dataset/sources.py`` as the modality label.

Usage::

    python scripts/fetch_medical_sources.py
    python scripts/fetch_medical_sources.py --count 50   # 50 per modality
    python scripts/fetch_medical_sources.py --split train val test

The ``--split`` argument controls which MedMNIST splits to pull from.  The
default is ``train`` only (the largest split, fewest duplicates with the test
images a model would be evaluated on).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from denoising.dataset.sources import write_gray_image  # noqa: E402
from denoising.logging_utils import get_logger  # noqa: E402

_LOG = get_logger(__name__)

# (MedMNIST dataset name, output subdirectory under data/raw/)
_DATASETS: list[tuple[str, str]] = [
    ("breastmnist", "ultrasound"),
    ("octmnist",    "mri"),
    ("chestmnist",  "ct"),
]


def _download_npz(url: str, dest: Path) -> bool:
    """Download *url* to *dest*; return True on success."""
    import hashlib
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _LOG.info("  downloading %s", url)
        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        return True
    except Exception as exc:
        _LOG.error("  download failed: %s", exc)
        if dest.with_suffix(".tmp").exists():
            dest.with_suffix(".tmp").unlink()
        return False


def _fetch_dataset(
    name: str,
    out_dir: Path,
    *,
    count: int,
    splits: list[str],
    size: int = 224,
) -> int:
    """Download *name* from MedMNIST via direct .npz download (no torch needed).

    Returns the number of images written.
    """
    try:
        from medmnist import INFO
    except ImportError:
        _LOG.error("medmnist is not installed — run:  pip install medmnist")
        return 0

    if name not in INFO:
        _LOG.error("unknown MedMNIST dataset %r", name)
        return 0

    info = INFO[name]
    size_key = "url" if size == 28 else f"url_{size}"
    url = info.get(size_key) or info.get("url_224") or info.get("url")
    if not url:
        _LOG.error("no download URL for %s at size %d", name, size)
        return 0

    # Cache the npz alongside the project's data directory.
    cache_dir = PROJECT_ROOT / "data" / ".medmnist_cache"
    npz_name = f"{name}_{size}.npz"
    npz_path = cache_dir / npz_name
    if not npz_path.exists():
        if not _download_npz(url, npz_path):
            return 0

    try:
        data = np.load(str(npz_path))
    except Exception as exc:
        _LOG.error("could not load %s: %s", npz_path, exc)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for split in splits:
        key = f"{split}_images"
        if key not in data:
            _LOG.warning("split %r not found in %s (keys: %s)", split, npz_name, list(data.keys()))
            continue

        images = data[key]  # shape: (N, H, W) or (N, H, W, C)
        remaining = count

        for idx in range(len(images)):
            if remaining <= 0:
                break
            fname = f"{name}_{split}_{idx:05d}.png"
            dest = out_dir / fname
            if dest.exists():
                remaining -= 1
                continue

            arr = images[idx]
            # Strip channel dim if present; convert RGB to gray.
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            elif arr.ndim == 3 and arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            elif arr.ndim == 3 and arr.shape[0] in (1, 3):
                # channels-first
                arr = arr[0] if arr.shape[0] == 1 else cv2.cvtColor(
                    np.transpose(arr, (1, 2, 0)), cv2.COLOR_RGB2GRAY
                )
            arr = arr.astype(np.uint8)

            write_gray_image(dest, arr)
            remaining -= 1
            written += 1

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_medical_sources",
        description="Download MedMNIST images into data/raw/{ultrasound,mri,ct}/.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="maximum images to save per modality (default: 100)",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train"],
        dest="splits",
        metavar="SPLIT",
        help="MedMNIST splits to pull from: train val test (default: train)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="destination root (default: data/raw)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="image size in pixels (default: 224, matches configs/dataset.yaml)",
    )
    args = parser.parse_args(argv)

    total = 0
    for ds_name, subdir in _DATASETS:
        out_dir = args.raw_dir / subdir
        _LOG.info("fetching %s → %s", ds_name, out_dir)
        n = _fetch_dataset(
            ds_name, out_dir, count=args.count, splits=args.splits, size=args.size
        )
        _LOG.info("  wrote %d image(s)", n)
        total += n

    _LOG.info("done — %d image(s) total", total)
    if total == 0:
        print(
            "\nNo images were written. If medmnist is not installed run:\n"
            "    pip install medmnist\n"
            "then re-run this script.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
