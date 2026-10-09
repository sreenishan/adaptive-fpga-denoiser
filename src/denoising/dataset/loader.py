"""Training-time manifest loading for the noise classifier dataset.

Reads the manifest CSV produced by :func:`~denoising.dataset.generate.generate_dataset`
and computes per-sample class weights for balanced training.

This module has no top-level torch dependency so it can be imported on machines
without PyTorch installed; class-weight computation uses only numpy.

Typical use (inside a training script that already imports torch)::

    from denoising.dataset.loader import load_manifest, sample_class_weights

    df   = load_manifest("data/processed/manifest.csv")
    wts  = sample_class_weights(df, split="train")
    # wts is a plain list[float] — pass directly to WeightedRandomSampler
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..config import CLASSES, PROJECT_ROOT

if TYPE_CHECKING:
    pass

__all__ = [
    "load_manifest",
    "split_counts",
    "sample_class_weights",
]

# Number of classes must match the config tuple so weights are always aligned.
_NUM_CLASSES = len(CLASSES)


def load_manifest(
    path: Path | str,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Read a manifest CSV and return it as a :class:`~pandas.DataFrame`.

    Args:
        path: Path to ``manifest.csv`` as written by the dataset generator.
        root: If given, prepend to any relative ``path`` column entries so
              :func:`load_image` can open them without knowing the dataset
              root.  Ignored when ``None``; absolute paths in the manifest
              are never changed.

    Returns:
        DataFrame with at least the columns ``path``, ``split``, ``label``,
        ``source_id``, ``noise_type``.  The ``label`` column is coerced to
        ``int``.
    """
    path = Path(path)
    df = pd.read_csv(path)

    required = {"path", "split", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"manifest {path} is missing required columns: {sorted(missing)}"
        )

    df["label"] = df["label"].astype(int)

    if root is not None:
        root = Path(root)
        # Only resolve entries that are not already absolute.
        mask = df["path"].apply(lambda p: not Path(p).is_absolute())
        df.loc[mask, "path"] = df.loc[mask, "path"].apply(
            lambda p: str(root / p)
        )

    return df


def split_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a ``(split, label, noise_type, count)`` summary table.

    Useful for a quick sanity-check that every split contains every class and
    that the class imbalance matches the expected 1:3 ratio.

    Args:
        df: A manifest DataFrame as returned by :func:`load_manifest`.

    Returns:
        DataFrame with columns ``split``, ``label``, ``noise_type``, ``count``,
        sorted by split then label.
    """
    group_cols = ["split", "label"]
    if "noise_type" in df.columns:
        group_cols.append("noise_type")
    summary = (
        df.groupby(group_cols, sort=True)
        .size()
        .reset_index(name="count")
    )
    return summary


def sample_class_weights(
    df: pd.DataFrame,
    split: str = "train",
) -> list[float]:
    """Compute a per-sample weight for balanced class sampling.

    Each sample receives ``total_samples / (num_classes * class_count)`` so
    that every class contributes equally in expectation when used with
    :class:`torch.utils.data.WeightedRandomSampler`.

    The returned list is parallel to the rows of ``df[df["split"] == split]``
    in their original order.  Pass it directly to ``WeightedRandomSampler``::

        from torch.utils.data import WeightedRandomSampler
        weights = sample_class_weights(df, "train")
        sampler = WeightedRandomSampler(weights, num_samples=len(weights))

    Args:
        df:    Manifest DataFrame (all splits; this function filters by split).
        split: Which split to compute weights for; must be one of
               ``"train"``, ``"val"``, ``"test"``.

    Returns:
        ``list[float]`` of length equal to the number of rows in *split*.

    Raises:
        ValueError: If *split* contains no rows.
    """
    rows = df[df["split"] == split]
    if rows.empty:
        raise ValueError(f"no rows for split={split!r} in manifest")

    labels = rows["label"].to_numpy(dtype=int)
    n = len(labels)

    # Count per class; clamp to 1 to avoid division by zero for absent classes.
    counts = np.zeros(_NUM_CLASSES, dtype=np.float64)
    for lbl in labels:
        counts[lbl] += 1
    counts = np.maximum(counts, 1.0)

    # Weight per class: inversely proportional to count.
    class_w = n / (_NUM_CLASSES * counts)

    return [float(class_w[lbl]) for lbl in labels]
