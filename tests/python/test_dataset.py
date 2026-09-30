"""Dataset generation, splitting and manifest (spec sections 8, 9 and 41)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from denoising import config as cfg
from denoising.dataset import (
    MANIFEST_COLUMNS,
    SPLITS,
    assign_splits,
    class_counts,
    generate_dataset,
    load_sources,
    plan_dataset,
    render_sample,
    sample_seed,
    synthetic_sources,
)
from denoising.dataset.cli import main
from denoising.dataset.sources import read_gray_image, write_gray_image

SOURCE_COUNT = 6


@pytest.fixture
def dataset_config(tmp_path: Path) -> cfg.DatasetConfig:
    """The shipped configuration, pointed at a temporary directory and shrunk
    to 32x32 so a full generation runs in well under a second."""
    raw = dict(cfg.load_yaml(cfg.CONFIG_DIR / "dataset.yaml"))
    raw["image"] = {"width": 32, "height": 32, "grayscale": True}
    raw["paths"] = {
        "raw_dir": "raw",
        "generated_dir": "generated",
        "manifest": "generated/manifest.csv",
    }
    return cfg.DatasetConfig.from_mapping(raw, root=tmp_path)


@pytest.fixture
def sources(dataset_config: cfg.DatasetConfig):
    return synthetic_sources(SOURCE_COUNT, dataset_config.image, seed=1)


# Splitting: the leakage guarantee.


def test_every_source_lands_in_exactly_one_split(dataset_config: cfg.DatasetConfig) -> None:
    ids = [f"src_{i:03d}" for i in range(20)]
    assignment = assign_splits(ids, dataset_config)
    assert set(assignment) == set(ids)
    assert set(assignment.values()) <= set(SPLITS)


def test_split_counts_sum_to_the_source_count(dataset_config: cfg.DatasetConfig) -> None:
    ids = [f"src_{i:03d}" for i in range(37)]
    assignment = assign_splits(ids, dataset_config)
    counts = {split: sum(1 for v in assignment.values() if v == split) for split in SPLITS}
    assert sum(counts.values()) == 37
    assert all(count >= 1 for count in counts.values())


def test_split_proportions_follow_the_configuration(dataset_config: cfg.DatasetConfig) -> None:
    ids = [f"src_{i:04d}" for i in range(200)]
    assignment = assign_splits(ids, dataset_config)
    train = sum(1 for v in assignment.values() if v == "train")
    assert train / 200 == pytest.approx(dataset_config.split.train_ratio, abs=0.02)


def test_split_assignment_is_deterministic(dataset_config: cfg.DatasetConfig) -> None:
    ids = [f"src_{i:03d}" for i in range(20)]
    assert assign_splits(ids, dataset_config) == assign_splits(ids, dataset_config)


def test_split_assignment_ignores_the_input_order(dataset_config: cfg.DatasetConfig) -> None:
    """The order files come off the filesystem must not decide the split."""
    ids = [f"src_{i:03d}" for i in range(20)]
    assert assign_splits(ids, dataset_config) == assign_splits(list(reversed(ids)), dataset_config)


def test_a_different_seed_gives_a_different_assignment(
    dataset_config: cfg.DatasetConfig, tmp_path: Path
) -> None:
    raw = dict(cfg.load_yaml(cfg.CONFIG_DIR / "dataset.yaml"))
    raw["split"] = dict(raw["split"], seed=999)
    other = cfg.DatasetConfig.from_mapping(raw, root=tmp_path)
    ids = [f"src_{i:03d}" for i in range(20)]
    assert assign_splits(ids, dataset_config) != assign_splits(ids, other)


def test_duplicate_source_ids_are_rejected(dataset_config: cfg.DatasetConfig) -> None:
    with pytest.raises(ValueError, match="duplicates"):
        assign_splits(["a", "b", "b"], dataset_config)


def test_too_few_sources_is_an_error(dataset_config: cfg.DatasetConfig) -> None:
    """Two sources cannot fill three splits, and an empty test set does not
    report an error — it reports nothing at all."""
    with pytest.raises(ValueError, match="at least 3 sources"):
        assign_splits(["a", "b"], dataset_config)


# Seeds.


def test_sample_seed_is_stable() -> None:
    """The constant is a measured output, not a chosen one. BLAKE2b is stable
    across Python versions, so this pins the derivation itself rather than an
    implementation detail of the hash: if the recipe changes, every seed in
    every existing manifest stops describing the image beside it."""
    assert sample_seed(42, "src_001", "gaussian", 1) == sample_seed(42, "src_001", "gaussian", 1)
    assert sample_seed(42, "src_001", "gaussian", 1) == 2644677424


def test_sample_seeds_differ_across_class_level_and_source() -> None:
    base = sample_seed(42, "src_001", "gaussian", 0)
    assert base != sample_seed(42, "src_001", "gaussian", 1)
    assert base != sample_seed(42, "src_001", "speckle", 0)
    assert base != sample_seed(42, "src_002", "gaussian", 0)
    assert base != sample_seed(43, "src_001", "gaussian", 0)


def test_seeds_do_not_shift_when_sources_are_added(
    dataset_config: cfg.DatasetConfig,
) -> None:
    """A counter-based seed would renumber every later sample when one source
    is inserted, silently changing images already generated."""
    first = {s.relative_path: s.seed for s in plan_dataset([f"s{i}" for i in range(5)], dataset_config)}
    second = {s.relative_path: s.seed for s in plan_dataset([f"s{i}" for i in range(6)], dataset_config)}
    shared = set(first) & set(second)
    assert shared
    for path in shared:
        assert first[path] == second[path]


# The plan.


def test_plan_covers_every_class_and_intensity(dataset_config: cfg.DatasetConfig) -> None:
    ids = [f"src_{i:03d}" for i in range(SOURCE_COUNT)]
    planned = plan_dataset(ids, dataset_config)
    noise = dataset_config.noise
    expected = {
        "clean": SOURCE_COUNT,
        "salt_pepper": SOURCE_COUNT * len(noise.salt_pepper.amounts),
        "gaussian": SOURCE_COUNT * len(noise.gaussian.sigmas),
        "speckle": SOURCE_COUNT * len(noise.speckle.variances),
    }
    assert class_counts(planned) == expected
    assert len(planned) == sum(expected.values())


def test_plan_labels_match_the_class_order(dataset_config: cfg.DatasetConfig) -> None:
    for sample in plan_dataset([f"s{i}" for i in range(4)], dataset_config):
        assert sample.label == cfg.CLASSES.index(sample.noise_type)


def test_plan_paths_are_unique_and_correctly_placed(dataset_config: cfg.DatasetConfig) -> None:
    planned = plan_dataset([f"s{i}" for i in range(SOURCE_COUNT)], dataset_config)
    paths = [sample.relative_path for sample in planned]
    assert len(set(paths)) == len(paths)
    for sample in planned:
        assert sample.relative_path.startswith(f"{sample.split}/{sample.noise_type}/")
        assert sample.relative_path.endswith(".png")


def test_all_samples_of_one_source_share_its_split(dataset_config: cfg.DatasetConfig) -> None:
    """The leakage guarantee, checked on the plan rather than on the files."""
    planned = plan_dataset([f"s{i}" for i in range(20)], dataset_config)
    splits_by_source: dict[str, set[str]] = {}
    for sample in planned:
        splits_by_source.setdefault(sample.source_id, set()).add(sample.split)
    assert all(len(found) == 1 for found in splits_by_source.values())


def test_clean_samples_carry_no_noise_parameter(dataset_config: cfg.DatasetConfig) -> None:
    planned = plan_dataset(["s0", "s1", "s2"], dataset_config)
    clean = [s for s in planned if s.noise_type == "clean"]
    assert clean
    assert all(s.noise_parameter is None and s.noise_parameters == {} for s in clean)


# Rendering.


def test_render_clean_returns_the_source_unchanged(
    dataset_config: cfg.DatasetConfig, sources
) -> None:
    planned = plan_dataset([s.source_id for s in sources], dataset_config)
    clean = next(s for s in planned if s.noise_type == "clean")
    source = next(s for s in sources if s.source_id == clean.source_id)
    assert np.array_equal(render_sample(clean, source.image), source.image)


def test_render_noisy_changes_pixels_and_keeps_the_shape(
    dataset_config: cfg.DatasetConfig, sources
) -> None:
    planned = plan_dataset([s.source_id for s in sources], dataset_config)
    for sample in planned:
        if sample.noise_type == "clean":
            continue
        source = next(s for s in sources if s.source_id == sample.source_id)
        out = render_sample(sample, source.image)
        assert out.shape == source.image.shape
        assert out.dtype == np.uint8
        assert not np.array_equal(out, source.image)


# Synthetic sources.


def test_synthetic_sources_are_deterministic(dataset_config: cfg.DatasetConfig) -> None:
    first = synthetic_sources(4, dataset_config.image, seed=7)
    second = synthetic_sources(4, dataset_config.image, seed=7)
    for a, b in zip(first, second):
        assert a.source_id == b.source_id
        assert np.array_equal(a.image, b.image)


def test_synthetic_sources_do_not_shift_when_more_are_asked_for(
    dataset_config: cfg.DatasetConfig,
) -> None:
    four = synthetic_sources(4, dataset_config.image, seed=7)
    six = synthetic_sources(6, dataset_config.image, seed=7)
    for a, b in zip(four, six):
        assert np.array_equal(a.image, b.image)


def test_synthetic_sources_have_structure(dataset_config: cfg.DatasetConfig) -> None:
    """A uniform source would make every noise class trivially separable."""
    for source in synthetic_sources(5, dataset_config.image, seed=3):
        assert source.image.dtype == np.uint8
        assert source.image.shape == (dataset_config.image.height, dataset_config.image.width)
        assert source.image.std() > 5.0
        assert source.origin == "synthetic"


def test_synthetic_count_must_be_positive(dataset_config: cfg.DatasetConfig) -> None:
    with pytest.raises(ValueError, match="count"):
        synthetic_sources(0, dataset_config.image, seed=1)


# Loading raw sources.


def test_load_sources_reads_resizes_and_greys(dataset_config: cfg.DatasetConfig, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_gray_image(raw / "a.png", np.full((64, 48), 90, dtype=np.uint8))
    write_gray_image(raw / "nested" / "b.png", np.full((16, 16), 200, dtype=np.uint8))
    loaded = load_sources(raw, dataset_config.image)
    assert [s.source_id for s in loaded] == ["a", "nested_b"]
    for source in loaded:
        assert source.image.shape == (32, 32)
        assert source.image.dtype == np.uint8


def test_load_sources_skips_files_that_are_not_images(
    dataset_config: cfg.DatasetConfig, tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_gray_image(raw / "good.png", np.full((32, 32), 10, dtype=np.uint8))
    (raw / "broken.png").write_bytes(b"not an image")
    (raw / "notes.txt").write_text("ignored", encoding="utf-8")
    assert [s.source_id for s in load_sources(raw, dataset_config.image)] == ["good"]


def test_load_sources_on_a_missing_directory_is_empty(
    dataset_config: cfg.DatasetConfig, tmp_path: Path
) -> None:
    assert load_sources(tmp_path / "absent", dataset_config.image) == []


# End to end.


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generate_writes_images_sidecars_and_manifest(
    dataset_config: cfg.DatasetConfig, sources
) -> None:
    summary = generate_dataset(sources, dataset_config)

    rows = _read_manifest(summary.manifest)
    assert list(rows[0]) == list(MANIFEST_COLUMNS)
    assert len(rows) == summary.samples

    images = sorted(summary.root.rglob("*.png"))
    sidecars = sorted(summary.root.rglob("*.json"))
    assert len(images) == summary.samples
    assert len(sidecars) == summary.samples


def test_generated_images_are_the_configured_size(
    dataset_config: cfg.DatasetConfig, sources
) -> None:
    summary = generate_dataset(sources, dataset_config)
    for path in summary.root.rglob("*.png"):
        image = read_gray_image(path)
        assert image is not None
        assert image.shape == (dataset_config.image.height, dataset_config.image.width)
        assert image.dtype == np.uint8


def test_manifest_and_sidecars_agree(dataset_config: cfg.DatasetConfig, sources) -> None:
    """They are written from one record in one pass; this fails the moment
    somebody gives them independent sources of truth."""
    summary = generate_dataset(sources, dataset_config)
    for row in _read_manifest(summary.manifest):
        image_path = cfg.PROJECT_ROOT / row["path"]
        if not image_path.exists():
            image_path = summary.root / row["path"]
        sidecar = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
        assert sidecar["class"] == row["noise_type"]
        assert sidecar["label"] == int(row["label"])
        assert sidecar["split"] == row["split"]
        assert sidecar["source_id"] == row["source_id"]
        assert sidecar["seed"] == int(row["seed"])


def test_no_source_appears_in_two_splits_on_disk(
    dataset_config: cfg.DatasetConfig, sources
) -> None:
    summary = generate_dataset(sources, dataset_config)
    seen: dict[str, str] = {}
    for row in _read_manifest(summary.manifest):
        seen.setdefault(row["source_id"], row["split"])
        assert seen[row["source_id"]] == row["split"]


def test_manifest_paths_are_relative(dataset_config: cfg.DatasetConfig, sources) -> None:
    summary = generate_dataset(sources, dataset_config)
    for row in _read_manifest(summary.manifest):
        assert not Path(row["path"]).is_absolute()
        assert "\\" not in row["path"]


def test_generation_is_reproducible(dataset_config: cfg.DatasetConfig, sources, tmp_path: Path) -> None:
    """Same sources, same config, same bytes."""
    first = generate_dataset(sources, dataset_config, root=tmp_path / "one", manifest=tmp_path / "one.csv")
    second = generate_dataset(sources, dataset_config, root=tmp_path / "two", manifest=tmp_path / "two.csv")
    for path in sorted(first.root.rglob("*.png")):
        twin = second.root / path.relative_to(first.root)
        assert path.read_bytes() == twin.read_bytes()


def test_generation_refuses_to_merge_into_an_existing_dataset(
    dataset_config: cfg.DatasetConfig, sources
) -> None:
    generate_dataset(sources, dataset_config)
    with pytest.raises(FileExistsError, match="overwrite"):
        generate_dataset(sources, dataset_config)


def test_overwrite_replaces_the_previous_run(dataset_config: cfg.DatasetConfig, sources) -> None:
    first = generate_dataset(sources, dataset_config)
    stale = first.root / "train" / "clean" / "stale__clean.png"
    write_gray_image(stale, np.zeros((32, 32), dtype=np.uint8))
    second = generate_dataset(sources, dataset_config, overwrite=True)
    assert not stale.exists()
    assert len(list(second.root.rglob("*.png"))) == second.samples


def test_generation_needs_sources(dataset_config: cfg.DatasetConfig) -> None:
    with pytest.raises(ValueError, match="no source images"):
        generate_dataset([], dataset_config)


# CLI.


def test_cli_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], dataset_config: cfg.DatasetConfig
) -> None:
    assert main(["--synthetic", "5", "--dry-run", "--output", str(tmp_path / "out")]) == 0
    assert "planned samples" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_cli_generates_a_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    code = main(
        [
            "--synthetic",
            "4",
            "--output",
            str(out),
            "--manifest",
            str(tmp_path / "manifest.csv"),
        ]
    )
    assert code == 0
    assert "samples" in capsys.readouterr().out
    assert (tmp_path / "manifest.csv").is_file()
    assert list(out.rglob("*.png"))


def test_cli_refuses_to_invent_sources_when_raw_is_empty(tmp_path: Path) -> None:
    """An empty data/raw must not silently become a synthetic dataset."""
    empty = tmp_path / "raw"
    empty.mkdir()
    assert main(["--raw-dir", str(empty), "--output", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()


def test_cli_reports_a_bad_configuration(tmp_path: Path) -> None:
    broken = tmp_path / "dataset.yaml"
    broken.write_text("image: {}\n", encoding="utf-8")
    assert main(["--config", str(broken), "--synthetic", "3"]) == 1
