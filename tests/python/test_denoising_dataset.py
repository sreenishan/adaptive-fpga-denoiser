"""Denoising dataset generation tests.

These tests verify the paired dataset generator without requiring PyTorch:
- Split assignment keeps all variants of one source in the same split.
- Seed derivation is deterministic (same inputs → same seed).
- Planned sample count = sources × noise_types × levels.
- Manifest columns match the spec.
- Generated files exist on disk and the manifest records them.
- Overwrite guard raises when dataset exists and overwrite=False.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from denoising.config import load_denoising_dataset_config
from denoising.dataset.denoising_generate import (
    DENOISING_MANIFEST_COLUMNS,
    denoising_sample_seed,
    generate_denoising_dataset,
    plan_denoising_dataset,
)
from denoising.dataset.sources import synthetic_sources


@pytest.fixture()
def config():
    return load_denoising_dataset_config()


@pytest.fixture()
def sources(config):
    return synthetic_sources(9, config.image, config.split.seed)


@pytest.fixture()
def planned(sources, config):
    ids = [s.source_id for s in sources]
    return plan_denoising_dataset(ids, config)


# ─── Plan-level tests (no file I/O) ──────────────────────────────────────────


def test_planned_count(planned, sources):
    expected = len(sources) * 3 * 4  # 3 types × 4 levels
    assert len(planned) == expected


def test_source_level_split(planned):
    """All 12 variants of each source land in the same split."""
    split_by_source: dict[str, set[str]] = {}
    for s in planned:
        split_by_source.setdefault(s.source_id, set()).add(s.split)
    for sid, splits in split_by_source.items():
        assert len(splits) == 1, f"{sid} straddles splits: {splits}"


def test_seed_determinism(planned, config):
    for s in planned:
        seed = denoising_sample_seed(config.split.seed, s.source_id, s.noise_type, s.level_index)
        assert seed == s.seed


def test_level_distribution(planned):
    by_level = Counter(s.noise_level for s in planned)
    assert set(by_level.keys()) == {5, 10, 15, 20}
    counts = list(by_level.values())
    assert len(set(counts)) == 1, f"unequal level counts: {by_level}"


def test_type_distribution(planned):
    by_type = Counter(s.noise_type for s in planned)
    assert set(by_type.keys()) == {"salt_pepper", "gaussian", "speckle"}


def test_noise_parameters_non_zero(planned):
    for s in planned:
        assert s.noise_parameter > 0, f"{s.noise_type} level {s.noise_level} has zero param"


def test_sp_parameters(planned):
    sp = [s for s in planned if s.noise_type == "salt_pepper"]
    amounts = sorted({s.noise_parameter for s in sp})
    assert amounts == pytest.approx([0.05, 0.10, 0.15, 0.20])


def test_gaussian_parameters(planned):
    g = [s for s in planned if s.noise_type == "gaussian"]
    sigmas = sorted({s.noise_parameter for s in g})
    assert sigmas == pytest.approx([0.05, 0.10, 0.15, 0.20])


def test_speckle_parameters(planned):
    sp = [s for s in planned if s.noise_type == "speckle"]
    variances = sorted({s.noise_parameter for s in sp})
    assert variances == pytest.approx([0.0025, 0.01, 0.0225, 0.04])


# ─── File generation tests ────────────────────────────────────────────────────


@pytest.fixture()
def tmpdir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_generate_creates_manifest(sources, config, tmpdir):
    manifest = tmpdir / "manifest.csv"
    summary = generate_denoising_dataset(
        sources, config, root=tmpdir, manifest=manifest, verbose=False
    )
    assert manifest.exists()
    assert summary.samples == len(sources) * 3 * 4


def test_manifest_columns(sources, config, tmpdir):
    manifest = tmpdir / "manifest.csv"
    generate_denoising_dataset(
        sources, config, root=tmpdir, manifest=manifest, verbose=False
    )
    with manifest.open() as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == DENOISING_MANIFEST_COLUMNS


def test_noisy_files_exist(sources, config, tmpdir):
    manifest = tmpdir / "manifest.csv"
    generate_denoising_dataset(
        sources, config, root=tmpdir, manifest=manifest, verbose=False
    )
    with manifest.open() as f:
        for row in csv.DictReader(f):
            noisy = tmpdir / row["noisy_path"]
            clean = tmpdir / row["clean_path"]
            assert noisy.exists(), f"missing: {noisy}"
            assert clean.exists(), f"missing: {clean}"


def test_overwrite_guard(sources, config, tmpdir):
    manifest = tmpdir / "manifest.csv"
    generate_denoising_dataset(
        sources, config, root=tmpdir, manifest=manifest, verbose=False
    )
    with pytest.raises(FileExistsError):
        generate_denoising_dataset(
            sources, config, root=tmpdir, manifest=manifest, overwrite=False, verbose=False
        )


def test_overwrite_replaces(sources, config, tmpdir):
    manifest = tmpdir / "manifest.csv"
    generate_denoising_dataset(
        sources, config, root=tmpdir, manifest=manifest, verbose=False
    )
    summary = generate_denoising_dataset(
        sources, config, root=tmpdir, manifest=manifest, overwrite=True, verbose=False
    )
    assert summary.samples == len(sources) * 3 * 4


def test_naming_convention(sources, config, tmpdir):
    manifest = tmpdir / "manifest.csv"
    generate_denoising_dataset(
        sources, config, root=tmpdir, manifest=manifest, verbose=False
    )
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    first = rows[0]
    noisy_name = Path(first["noisy_path"]).name
    # e.g. synthetic_0000__salt_pepper__5.png
    assert "__" in noisy_name
    parts = noisy_name.replace(".png", "").split("__")
    assert len(parts) == 3, f"unexpected name: {noisy_name}"
    assert parts[1] in {"salt_pepper", "gaussian", "speckle"}
    assert int(parts[2]) in {5, 10, 15, 20}
