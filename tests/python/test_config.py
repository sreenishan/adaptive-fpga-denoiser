"""Configuration loading and validation (spec section 41)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from denoising import config as cfg


def _raw(name: str) -> dict[str, Any]:
    """A mutable copy of a shipped configuration file."""
    return copy.deepcopy(dict(cfg.load_yaml(cfg.CONFIG_DIR / name)))


# The files that ship with the repository must load.


def test_shipped_dataset_config_loads() -> None:
    dataset = cfg.load_dataset_config()
    assert dataset.image.grayscale is True
    total = (
        dataset.split.train_ratio
        + dataset.split.validation_ratio
        + dataset.split.test_ratio
    )
    assert total == pytest.approx(1.0)
    # Multiple intensities, never a single fixed noise level (spec section 8).
    assert len(dataset.noise.salt_pepper.amounts) >= 3
    assert len(dataset.noise.gaussian.sigmas) >= 3
    assert len(dataset.noise.speckle.variances) >= 3


def test_shipped_training_config_loads() -> None:
    training = cfg.load_training_config()
    assert training.model.num_classes == len(cfg.CLASSES)
    assert training.optimizer in ("adam", "adamw", "sgd")
    assert training.early_stopping.patience >= 1


def test_shipped_inference_config_loads() -> None:
    inference = cfg.load_inference_config()
    assert 0.0 <= inference.confidence.threshold <= 1.0
    assert inference.confidence.fallback in cfg.FALLBACK_FILTERS
    assert inference.filters.median.kernel_size == 3
    assert inference.filters.gaussian.kernel_size == 3
    assert inference.filters.wiener.kernel_size == 3


def test_shipped_hardware_config_loads() -> None:
    hardware = cfg.load_hardware_config()
    assert hardware.stream.pixel_width == 8
    assert hardware.stream.boundary_policy == "replicate"
    # Bit-exact is the requirement for the two exact filters (spec section 29).
    assert hardware.simulation.max_abs_error["median"] == 0
    assert hardware.simulation.max_abs_error["gaussian"] == 0


def test_software_and_rtl_boundary_policies_agree() -> None:
    """A window generator and its golden reference disagreeing at the edges
    shows up as a verification failure with no obvious cause."""
    software = cfg.load_inference_config().filters.boundary_mode
    hardware = cfg.load_hardware_config().stream.boundary_policy
    assert software == hardware


# Honesty: nothing unmeasured is reported as a number.


def test_unconfigured_synthesis_is_none_not_zero() -> None:
    synthesis = cfg.load_hardware_config().synthesis
    assert synthesis.vendor is None
    assert synthesis.device is None
    assert synthesis.clock_mhz is None
    assert synthesis.tool_version is None
    assert synthesis.configured is False


def test_wiener_noise_variance_may_be_unspecified() -> None:
    """null means estimate it, which is not the same claim as 0.0."""
    assert cfg.load_inference_config().filters.wiener.noise_variance is None


# Paths


def test_relative_paths_resolve_against_the_project_root() -> None:
    dataset = cfg.load_dataset_config()
    assert dataset.paths.manifest.is_absolute()
    expected_parent = (cfg.PROJECT_ROOT / "data" / "generated").resolve()
    assert dataset.paths.manifest.parent == expected_parent


def test_paths_resolve_against_a_supplied_root(tmp_path: Path) -> None:
    dataset = cfg.DatasetConfig.from_mapping(_raw("dataset.yaml"), root=tmp_path)
    assert dataset.paths.raw_dir == (tmp_path / "data" / "raw").resolve()


def test_no_absolute_paths_in_the_shipped_configs() -> None:
    for name in ("dataset.yaml", "training.yaml", "inference.yaml", "hardware.yaml"):
        for line in (cfg.CONFIG_DIR / name).read_text(encoding="utf-8").splitlines():
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            value = line.split(":", 1)[1].strip()
            assert not value.startswith("/"), f"{name}: absolute path {value!r}"
            windows_drive = len(value) > 2 and value[1] == ":"
            assert not windows_drive, f"{name}: absolute path {value!r}"


# Rejections. Each error must name the key that is wrong.


def test_split_ratios_must_sum_to_one() -> None:
    data = _raw("dataset.yaml")
    data["split"]["test_ratio"] = 0.25
    with pytest.raises(cfg.ConfigError, match="ratios must sum"):
        cfg.DatasetConfig.from_mapping(data)


def test_missing_key_is_named() -> None:
    data = _raw("dataset.yaml")
    del data["split"]["seed"]
    with pytest.raises(cfg.ConfigError, match="split.seed"):
        cfg.DatasetConfig.from_mapping(data)


@pytest.mark.parametrize("amount", [0.0, 1.5, -0.1])
def test_salt_pepper_amount_out_of_range_is_rejected(amount: float) -> None:
    data = _raw("dataset.yaml")
    data["noise"]["salt_pepper"]["amounts"] = [amount]
    with pytest.raises(cfg.ConfigError, match="amounts"):
        cfg.DatasetConfig.from_mapping(data)


def test_empty_intensity_list_is_rejected() -> None:
    data = _raw("dataset.yaml")
    data["noise"]["gaussian"]["sigmas"] = []
    with pytest.raises(cfg.ConfigError, match="sigmas"):
        cfg.DatasetConfig.from_mapping(data)


def test_num_classes_must_match_the_class_list() -> None:
    data = _raw("training.yaml")
    data["model"]["num_classes"] = 3
    with pytest.raises(cfg.ConfigError, match="num_classes"):
        cfg.TrainingConfig.from_mapping(data)


def test_unknown_optimizer_is_rejected() -> None:
    data = _raw("training.yaml")
    data["training"]["optimizer"] = "rmsprop"
    with pytest.raises(cfg.ConfigError, match="optimizer"):
        cfg.TrainingConfig.from_mapping(data)


def test_scheduler_may_be_null_but_not_arbitrary() -> None:
    data = _raw("training.yaml")
    data["training"]["scheduler"] = None
    assert cfg.TrainingConfig.from_mapping(data).scheduler is None
    data["training"]["scheduler"] = "magic"
    with pytest.raises(cfg.ConfigError, match="scheduler"):
        cfg.TrainingConfig.from_mapping(data)


def test_even_kernel_size_is_rejected() -> None:
    data = _raw("inference.yaml")
    data["filters"]["median"]["kernel_size"] = 4
    with pytest.raises(cfg.ConfigError, match="must be odd"):
        cfg.InferenceConfig.from_mapping(data)


def test_unknown_fallback_filter_is_rejected() -> None:
    data = _raw("inference.yaml")
    data["confidence"]["fallback"] = "denoise-really-hard"
    with pytest.raises(cfg.ConfigError, match="fallback"):
        cfg.InferenceConfig.from_mapping(data)


def test_confidence_threshold_outside_zero_to_one_is_rejected() -> None:
    data = _raw("inference.yaml")
    data["confidence"]["threshold"] = 1.4
    with pytest.raises(cfg.ConfigError, match="threshold"):
        cfg.InferenceConfig.from_mapping(data)


def test_unknown_boundary_mode_is_rejected() -> None:
    data = _raw("inference.yaml")
    data["filters"]["boundary_mode"] = "wrap"
    with pytest.raises(cfg.ConfigError, match="boundary_mode"):
        cfg.InferenceConfig.from_mapping(data)


def test_pixel_width_other_than_eight_is_rejected() -> None:
    """Only 8-bit grayscale is implemented; a 10-bit claim would be untrue."""
    data = _raw("hardware.yaml")
    data["stream"]["pixel_width"] = 10
    with pytest.raises(cfg.ConfigError, match="pixel_width"):
        cfg.HardwareConfig.from_mapping(data)


def test_unimplemented_boundary_policy_is_rejected() -> None:
    data = _raw("hardware.yaml")
    data["stream"]["boundary_policy"] = "zero"
    with pytest.raises(cfg.ConfigError, match="boundary_policy"):
        cfg.HardwareConfig.from_mapping(data)


def test_unknown_simulator_is_rejected() -> None:
    data = _raw("hardware.yaml")
    data["simulation"]["simulator"] = "modelsim"
    with pytest.raises(cfg.ConfigError, match="simulator"):
        cfg.HardwareConfig.from_mapping(data)


def test_booleans_are_not_accepted_as_numbers() -> None:
    data = _raw("dataset.yaml")
    data["image"]["width"] = True
    with pytest.raises(cfg.ConfigError, match="image.width"):
        cfg.DatasetConfig.from_mapping(data)


def test_section_of_the_wrong_type_is_rejected() -> None:
    data = _raw("dataset.yaml")
    data["noise"] = "lots"
    with pytest.raises(cfg.ConfigError, match="noise"):
        cfg.DatasetConfig.from_mapping(data)


# load_yaml


def test_missing_file_is_reported_as_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(cfg.ConfigError, match="not found"):
        cfg.load_yaml(tmp_path / "absent.yaml")


def test_empty_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(cfg.ConfigError, match="empty"):
        cfg.load_yaml(path)


def test_non_mapping_top_level_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError, match="mapping"):
        cfg.load_yaml(path)


def test_round_trip_through_a_written_file(tmp_path: Path) -> None:
    """A config written back out must load to the same values."""
    data: Mapping[str, Any] = _raw("dataset.yaml")
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(dict(data)), encoding="utf-8")
    assert cfg.load_dataset_config(path).noise == cfg.load_dataset_config().noise


# Frozen dataclasses


def test_configs_are_immutable() -> None:
    dataset = cfg.load_dataset_config()
    with pytest.raises(Exception):
        dataset.image.width = 32  # type: ignore[misc]
