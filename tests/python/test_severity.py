"""Noise severity, adaptive median and severity-driven filter selection (stage 4-5)."""

from __future__ import annotations

import numpy as np
import pytest

from denoising import config as cfg
from denoising.filters import (
    ALL_FILTERS,
    FILTER_FOR_CLASS,
    FILTERS,
    SEVERITY_LEVELS,
    SEVERITY_POLICY,
    SOFTWARE_FILTERS,
    adaptive_median_filter,
    apply_filter,
    decide_filter,
    median_filter,
)
from denoising.metrics import calculate_psnr
from denoising.noise import add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise
from denoising.pipeline import process_image
from denoising.severity import estimate_severity, impulse_fraction, noise_sigma, speckle_ratio


@pytest.fixture
def inference() -> cfg.InferenceConfig:
    return cfg.load_inference_config()


@pytest.fixture
def scene() -> np.ndarray:
    """Smooth gradients plus a flat block — enough structure to fool a naive
    variance estimator, none of it noise."""
    yy, xx = np.mgrid[0:96, 0:96]
    image = (60 + xx * 1.2 + yy * 0.6).astype(np.uint8)
    image[30:60, 30:60] = 170
    return image


# Estimators ------------------------------------------------------------------


@pytest.mark.parametrize("amount", [0.02, 0.05, 0.10, 0.30])
def test_impulse_fraction_recovers_the_noise_amount(scene: np.ndarray, amount: float) -> None:
    noisy = add_salt_pepper_noise(scene, amount, seed=3)
    assert impulse_fraction(noisy) == pytest.approx(amount, rel=0.15)


def test_impulse_fraction_ignores_real_black_and_white_regions(
    inference: cfg.InferenceConfig,
) -> None:
    """A black square is image content. Counting extreme pixels alone would
    score its 596 pixels as heavy noise and pick the strongest filter.

    Not exactly zero: a convex corner has only 4 of 9 matching neighbours, so
    its median is the background and it counts — 8 pixels here. What must
    hold is that real shapes stay far below the first cut point."""
    image = np.full((48, 48), 128, np.uint8)
    image[10:30, 10:30] = 0
    image[30:44, 30:44] = 255
    fraction = impulse_fraction(image)
    assert fraction == pytest.approx(8 / image.size)
    assert fraction < inference.severity.salt_pepper.medium_from / 5


@pytest.mark.parametrize("sigma", [0.03, 0.06, 0.10])
def test_noise_sigma_recovers_gaussian_sigma(scene: np.ndarray, sigma: float) -> None:
    noisy = add_gaussian_noise(scene, 0.0, sigma, seed=4)
    assert noise_sigma(noisy) / 255 == pytest.approx(sigma, rel=0.10)


def test_noise_sigma_is_near_zero_on_smooth_structure(scene: np.ndarray) -> None:
    """Why Immerkaer and not local variance: a gradient is not noise."""
    assert noise_sigma(scene) < 1.5


@pytest.mark.parametrize("variance", [0.03, 0.06, 0.10])
def test_speckle_ratio_tracks_the_speckle_std(scene: np.ndarray, variance: float) -> None:
    noisy = add_speckle_noise(scene, variance, seed=5)
    assert speckle_ratio(noisy) == pytest.approx(variance**0.5, rel=0.15)


def test_estimators_reject_images_too_small_for_a_window() -> None:
    with pytest.raises(ValueError):
        noise_sigma(np.zeros((2, 5), np.uint8))


# Levels ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("noise_class", "levels", "make"),
    [
        ("salt_pepper", (0.02, 0.05, 0.10), lambda s, a: add_salt_pepper_noise(s, a, seed=6)),
        ("gaussian", (0.03, 0.06, 0.10), lambda s, a: add_gaussian_noise(s, 0.0, a, seed=6)),
        ("speckle", (0.03, 0.06, 0.10), lambda s, a: add_speckle_noise(s, a, seed=6)),
    ],
)
def test_dataset_levels_map_to_low_medium_high(
    scene: np.ndarray, inference: cfg.InferenceConfig, noise_class, levels, make
) -> None:
    got = [estimate_severity(make(scene, a), noise_class, inference.severity).level for a in levels]
    assert got == list(SEVERITY_LEVELS)


def test_clean_has_no_severity(scene: np.ndarray, inference: cfg.InferenceConfig) -> None:
    """No noise has no strength; a level here would measure nothing."""
    assert estimate_severity(scene, "clean", inference.severity) is None


def test_disabled_severity_returns_none(scene: np.ndarray, inference: cfg.InferenceConfig) -> None:
    off = cfg.SeverityConfig(
        enabled=False,
        salt_pepper=inference.severity.salt_pepper,
        gaussian=inference.severity.gaussian,
        speckle=inference.severity.speckle,
    )
    noisy = add_gaussian_noise(scene, 0.0, 0.1, seed=1)
    assert estimate_severity(noisy, "gaussian", off) is None


def test_a_band_with_no_medium_range_is_rejected() -> None:
    with pytest.raises(cfg.ConfigError):
        cfg.SeverityBand.from_mapping({"medium_from": 0.08, "high_from": 0.08}, "severity.gaussian")


# Adaptive median -------------------------------------------------------------


def test_adaptive_median_only_touches_impulses() -> None:
    """The property that separates it from a plain median: a pixel that is
    not the extreme of its window comes through exactly as it went in."""
    # A planar ramp: every pixel sits strictly between its window's min and
    # max, so the only local extremes in the noisy image are the impulses.
    yy, xx = np.mgrid[0:40, 0:40]
    image = (20 + 2 * xx + 2 * yy).astype(np.uint8)
    noisy = image.copy()
    rng = np.random.default_rng(0)
    hits = rng.choice(image.size, 30, replace=False)
    noisy.flat[hits[:15]] = 0
    noisy.flat[hits[15:]] = 255
    out = adaptive_median_filter(noisy)
    untouched = np.ones(image.size, bool)
    untouched[hits] = False
    untouched = untouched.reshape(image.shape)
    interior = np.zeros_like(untouched)
    interior[1:-1, 1:-1] = True
    # Away from the border and from the impulses, nothing moved.
    assert np.array_equal(out[untouched & interior], noisy[untouched & interior])
    assert not np.isin(out.flat[hits], (0, 255)).all()


def test_adaptive_median_beats_plain_median_at_high_density() -> None:
    yy, xx = np.mgrid[0:96, 0:96]
    clean = (80 + xx + yy // 2).astype(np.uint8)
    noisy = add_salt_pepper_noise(clean, 0.30, seed=2)
    plain = calculate_psnr(clean, median_filter(noisy, 3))
    adaptive = calculate_psnr(clean, adaptive_median_filter(noisy))
    assert adaptive > plain + 5.0


def test_adaptive_median_preserves_shape_dtype_and_input() -> None:
    noisy = add_salt_pepper_noise(np.full((20, 17), 90, np.uint8), 0.2, seed=1)
    before = noisy.copy()
    out = adaptive_median_filter(noisy)
    assert out.shape == noisy.shape and out.dtype == np.uint8
    assert np.array_equal(noisy, before)


def test_adaptive_median_skips_windows_larger_than_the_image() -> None:
    out = adaptive_median_filter(np.full((4, 9), 50, np.uint8), max_size=7)
    assert out.shape == (4, 9)


@pytest.mark.parametrize("bad", [2, 4, 1])
def test_adaptive_median_rejects_bad_sizes(bad: int) -> None:
    with pytest.raises(ValueError):
        adaptive_median_filter(np.zeros((10, 10), np.uint8), max_size=bad)


def test_apply_filter_dispatches_adaptive_median() -> None:
    noisy = add_salt_pepper_noise(np.full((16, 16), 100, np.uint8), 0.1, seed=0)
    assert np.array_equal(apply_filter(noisy, "adaptive_median"), adaptive_median_filter(noisy))


# Policy ----------------------------------------------------------------------


def test_software_filters_stay_out_of_the_hardware_code_space() -> None:
    """FILTERS is the RTL's 2-bit control-code space. A software-only filter
    in it would be assumed to run on the FPGA."""
    assert not set(SOFTWARE_FILTERS) & set(FILTERS)
    assert set(ALL_FILTERS) == set(FILTERS) | set(SOFTWARE_FILTERS)


def test_without_severity_the_mapping_is_exactly_the_old_one() -> None:
    for noise_class, mapped in FILTER_FOR_CLASS.items():
        decision = decide_filter(noise_class)
        assert (decision.filter_name, decision.passes, decision.severity) == (mapped, 1, None)


@pytest.mark.parametrize("noise_class", sorted(SEVERITY_POLICY))
@pytest.mark.parametrize("level", SEVERITY_LEVELS)
def test_severity_selects_the_policy_step(noise_class: str, level: str) -> None:
    decision = decide_filter(noise_class, severity=level)
    step = SEVERITY_POLICY[noise_class][level]
    assert (decision.filter_name, decision.passes, decision.severity) == (
        step.filter_name,
        step.passes,
        level,
    )


def test_clean_ignores_severity() -> None:
    decision = decide_filter("clean", severity="high")
    assert (decision.filter_name, decision.passes, decision.severity) == ("bypass", 1, None)


def test_the_fallback_ignores_severity() -> None:
    """An untrusted prediction cannot choose a strength either."""
    decision = decide_filter("salt_pepper", 0.2, threshold=0.6, fallback="bypass", severity="high")
    assert decision.used_fallback
    assert (decision.filter_name, decision.passes, decision.severity) == ("bypass", 1, None)
    assert decision.mapped_filter == "adaptive_median"


def test_unknown_severity_is_rejected() -> None:
    with pytest.raises(ValueError):
        decide_filter("gaussian", severity="extreme")


def test_hardware_status_is_reported_honestly() -> None:
    assert decide_filter("salt_pepper", severity="low").hardware == "rtl"
    assert decide_filter("gaussian", severity="low").hardware == "rtl_multipass"
    software = decide_filter("salt_pepper", severity="high")
    assert software.hardware == "software"
    # No borrowed 2'b01: the hardware would run a plain median instead.
    assert software.control_code is None


# Pipeline --------------------------------------------------------------------


def test_pipeline_reports_severity_and_applies_every_pass(
    scene: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    noisy = add_gaussian_noise(scene, 0.0, 0.03, seed=9)
    result = process_image(noisy, inference, noise_class="gaussian", reference=scene)
    assert result.severity is not None and result.severity.level == "low"
    assert result.decision.passes == 2
    expected = noisy
    for _ in range(2):
        expected = apply_filter(expected, "wiener", kernel_size=3, noise_variance=None)
    assert np.array_equal(result.output, expected)


def test_pipeline_severity_improves_on_the_fixed_mapping(
    scene: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    noisy = add_salt_pepper_noise(scene, 0.30, seed=11)
    adaptive = process_image(noisy, inference, noise_class="salt_pepper", reference=scene)
    fixed = median_filter(noisy, 3)
    assert adaptive.decision.filter_name == "adaptive_median"
    assert adaptive.metrics.psnr > calculate_psnr(scene, fixed) + 5.0


def test_pipeline_result_serialises_the_new_fields(
    scene: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    noisy = add_salt_pepper_noise(scene, 0.30, seed=11)
    record = process_image(noisy, inference, noise_class="salt_pepper").as_dict()
    assert record["hardware"] == "software"
    assert record["control_code"] is None
    assert record["severity"]["level"] == "high"
    assert record["severity"]["applied"] is True
