"""Metrics, preprocessing and the end-to-end pipeline (spec 10, 16, 17, 41)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from denoising import config as cfg
from denoising.dataset.sources import write_gray_image
from denoising.metrics import (
    ImageQuality,
    calculate_mse,
    calculate_psnr,
    calculate_quality,
    calculate_ssim,
)
from denoising.noise import add_gaussian_noise, add_salt_pepper_noise
from denoising.pipeline import process_image
from denoising.preprocessing import load_image, preprocess, to_grayscale, to_model_input


@pytest.fixture
def clean() -> np.ndarray:
    yy, xx = np.mgrid[0:32, 0:32]
    image = ((xx * 6 + yy * 3) % 256).astype(np.uint8)
    image[8:16, 8:16] = 210
    return image


@pytest.fixture
def inference() -> cfg.InferenceConfig:
    return cfg.load_inference_config()


# Metrics.


def test_identical_images_score_perfectly(clean: np.ndarray) -> None:
    quality = calculate_quality(clean, clean)
    assert quality.mse == 0.0
    assert quality.psnr == math.inf
    assert quality.ssim == pytest.approx(1.0)
    assert quality.identical is True


def test_psnr_is_infinite_not_a_large_number(clean: np.ndarray) -> None:
    """A finite stand-in would be a measurement-shaped value where the
    quantity is unbounded."""
    assert math.isinf(calculate_psnr(clean, clean))


def test_mse_matches_a_hand_computed_case() -> None:
    a = np.array([[10, 10], [10, 10]], dtype=np.uint8)
    b = np.array([[12, 10], [10, 6]], dtype=np.uint8)
    # differences 2, 0, 0, -4 -> (4 + 0 + 0 + 16) / 4 = 5
    assert calculate_mse(a, b) == pytest.approx(5.0)


def test_mse_does_not_wrap_around_on_uint8() -> None:
    """250 - 10 must be 240, not the 16 that uint8 subtraction gives."""
    a = np.full((4, 4), 250, dtype=np.uint8)
    b = np.full((4, 4), 10, dtype=np.uint8)
    assert calculate_mse(a, b) == pytest.approx(240.0**2)


def test_psnr_matches_the_formula() -> None:
    a = np.zeros((8, 8), dtype=np.uint8)
    b = np.full((8, 8), 1, dtype=np.uint8)
    assert calculate_psnr(a, b) == pytest.approx(10.0 * math.log10(255**2 / 1.0))


def test_psnr_falls_as_noise_grows(clean: np.ndarray) -> None:
    light = add_gaussian_noise(clean, 0.0, 0.02, seed=1)
    heavy = add_gaussian_noise(clean, 0.0, 0.20, seed=1)
    assert calculate_psnr(clean, light) > calculate_psnr(clean, heavy)


def test_ssim_falls_as_structure_is_destroyed(clean: np.ndarray) -> None:
    noisy = add_salt_pepper_noise(clean, 0.20, 0.5, seed=1)
    assert calculate_ssim(clean, noisy) < calculate_ssim(clean, clean)


def test_metrics_reject_a_shape_mismatch(clean: np.ndarray) -> None:
    with pytest.raises(ValueError, match="same shape"):
        calculate_mse(clean, clean[:16])


def test_ssim_needs_a_big_enough_image() -> None:
    tiny = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="7x7"):
        calculate_ssim(tiny, tiny)


def test_quality_is_immutable(clean: np.ndarray) -> None:
    quality = calculate_quality(clean, clean)
    assert isinstance(quality, ImageQuality)
    with pytest.raises(Exception):
        quality.mse = 1.0  # type: ignore[misc]


# Preprocessing.


def test_preprocess_resizes_to_the_configured_geometry(clean: np.ndarray) -> None:
    image_cfg = cfg.ImageConfig(width=64, height=48, grayscale=True)
    out = preprocess(clean, image_cfg)
    assert out.shape == (48, 64)
    assert out.dtype == np.uint8


def test_model_input_is_normalised_float32(clean: np.ndarray) -> None:
    image_cfg = cfg.ImageConfig(width=32, height=32, grayscale=True)
    tensor = to_model_input(clean, image_cfg)
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 1, 32, 32)
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_model_input_without_a_batch_axis(clean: np.ndarray) -> None:
    image_cfg = cfg.ImageConfig(width=32, height=32, grayscale=True)
    assert to_model_input(clean, image_cfg, add_batch=False).shape == (1, 32, 32)


def test_normalisation_maps_the_rails_exactly() -> None:
    image_cfg = cfg.ImageConfig(width=8, height=8, grayscale=True)
    assert to_model_input(np.zeros((8, 8), np.uint8), image_cfg).max() == 0.0
    assert to_model_input(np.full((8, 8), 255, np.uint8), image_cfg).min() == 1.0


def test_colour_is_reduced_with_luma_weights() -> None:
    red = np.zeros((4, 4, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    assert (to_grayscale(red) == 76).all()  # rint(0.299 * 255) = 76
    assert to_grayscale(np.zeros((4, 4), np.uint8)).shape == (4, 4)


def test_load_image_round_trips(tmp_path: Path, clean: np.ndarray) -> None:
    path = tmp_path / "image.png"
    write_gray_image(path, clean)
    assert np.array_equal(load_image(path), clean)


def test_load_image_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "absent.png")


def test_load_image_reports_a_file_that_is_not_an_image(tmp_path: Path) -> None:
    path = tmp_path / "notes.png"
    path.write_bytes(b"definitely not a png")
    with pytest.raises(ValueError, match="could not be decoded"):
        load_image(path)


# Pipeline.


class _StubClassifier:
    """Stands in for phase 6, so the pipeline can be tested before it exists."""

    def __init__(self, noise_class: str, confidence: float) -> None:
        self._class = noise_class
        self._confidence = confidence
        self.calls = 0

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        self.calls += 1
        return self._class, self._confidence


def test_manual_class_selects_the_mapped_filter(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    noisy = add_salt_pepper_noise(clean, 0.05, 0.5, seed=1)
    result = process_image(noisy, inference, noise_class="salt_pepper")
    assert result.selected_filter == "median"
    assert result.decision.control_code == 1
    assert result.output.shape == noisy.shape


def test_a_manual_choice_reports_no_confidence(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    """Reporting 1.0 would invent a measurement nobody made."""
    result = process_image(clean, inference, noise_class="clean")
    assert result.confidence is None
    assert result.as_dict()["confidence"] is None


def test_a_classifier_is_used_when_supplied(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    stub = _StubClassifier("speckle", 0.91)
    result = process_image(clean, inference, classifier=stub)
    assert stub.calls == 1
    assert result.noise_class == "speckle"
    assert result.confidence == pytest.approx(0.91)
    assert result.selected_filter == "wiener"


def test_low_confidence_falls_back_and_records_it(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    stub = _StubClassifier("speckle", 0.10)
    result = process_image(clean, inference, classifier=stub)
    assert result.decision.used_fallback is True
    assert result.selected_filter == inference.confidence.fallback
    assert result.decision.mapped_filter == "wiener"


def test_clean_input_is_passed_through_unchanged(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    result = process_image(clean, inference, noise_class="clean")
    assert result.selected_filter == "bypass"
    assert np.array_equal(result.output, clean)


def test_metrics_are_absent_without_a_reference(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    """The honesty rule: no clean original, no quality figures."""
    noisy = add_gaussian_noise(clean, 0.0, 0.08, seed=1)
    result = process_image(noisy, inference, noise_class="gaussian")
    assert result.metrics is None
    assert result.psnr_improvement is None
    assert result.metrics_note is not None
    assert "no clean reference" in result.metrics_note


def test_metrics_are_computed_against_a_reference(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    noisy = add_gaussian_noise(clean, 0.0, 0.08, seed=1)
    result = process_image(noisy, inference, noise_class="gaussian", reference=clean)
    assert result.metrics is not None
    assert result.noisy_metrics is not None
    assert result.metrics_note is None
    assert result.metrics.mse < result.noisy_metrics.mse


def test_the_right_filter_improves_psnr(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    """Measured, not assumed: this is the claim the whole project rests on."""
    noisy = add_salt_pepper_noise(clean, 0.08, 0.5, seed=3)
    result = process_image(noisy, inference, noise_class="salt_pepper", reference=clean)
    assert result.psnr_improvement is not None
    assert result.psnr_improvement > 0.0


def test_timings_are_recorded(clean: np.ndarray, inference: cfg.InferenceConfig) -> None:
    result = process_image(clean, inference, noise_class="gaussian")
    assert set(result.timings_ms) == {"preprocess", "classify", "filter", "total"}
    assert all(value >= 0.0 for value in result.timings_ms.values())


def test_resize_is_off_by_default(inference: cfg.InferenceConfig) -> None:
    """A caller submits an image and gets that image back, not a rescaled one."""
    odd = np.full((37, 53), 120, dtype=np.uint8)
    assert process_image(odd, inference, noise_class="clean").output.shape == (37, 53)
    resized = process_image(odd, inference, noise_class="clean", resize=True)
    assert resized.output.shape == (inference.image.height, inference.image.width)


def test_exactly_one_of_class_or_classifier_is_required(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        process_image(clean, inference)
    with pytest.raises(ValueError, match="exactly one"):
        process_image(
            clean, inference, noise_class="clean", classifier=_StubClassifier("clean", 1.0)
        )


def test_an_unknown_manual_class_is_rejected(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    with pytest.raises(ValueError, match="unknown noise class"):
        process_image(clean, inference, noise_class="blurry")


def test_a_classifier_returning_nonsense_is_rejected(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    with pytest.raises(ValueError, match="unknown class"):
        process_image(clean, inference, classifier=_StubClassifier("banana", 0.99))


def test_a_mismatched_reference_is_rejected(
    clean: np.ndarray, inference: cfg.InferenceConfig
) -> None:
    with pytest.raises(ValueError, match="does not match"):
        process_image(clean, inference, noise_class="clean", reference=clean[:16])


def test_as_dict_is_json_shaped(clean: np.ndarray, inference: cfg.InferenceConfig) -> None:
    result = process_image(clean, inference, noise_class="clean", reference=clean)
    payload = result.as_dict()
    assert payload["noise_class"] == "clean"
    assert payload["selected_filter"] == "bypass"
    assert payload["control_code"] == 0
    assert payload["metrics"]["mse"] == 0.0
    assert math.isinf(payload["metrics"]["psnr"])
