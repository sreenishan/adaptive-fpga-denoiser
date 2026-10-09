"""Rician noise generator (spec section 7.4)."""

from __future__ import annotations

import numpy as np
import pytest

from denoising.noise import add_rician_noise


@pytest.fixture
def mid_gray() -> np.ndarray:
    return np.full((64, 64), 128, dtype=np.uint8)


def test_shape_and_dtype_preserved(mid_gray: np.ndarray) -> None:
    out = add_rician_noise(mid_gray, sigma=0.05, seed=1)
    assert out.shape == mid_gray.shape
    assert out.dtype == np.uint8


def test_input_not_modified(mid_gray: np.ndarray) -> None:
    original = mid_gray.copy()
    add_rician_noise(mid_gray, sigma=0.05, seed=1)
    assert np.array_equal(mid_gray, original)


def test_same_seed_gives_same_output(mid_gray: np.ndarray) -> None:
    a = add_rician_noise(mid_gray, sigma=0.05, seed=42)
    b = add_rician_noise(mid_gray, sigma=0.05, seed=42)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_outputs(mid_gray: np.ndarray) -> None:
    a = add_rician_noise(mid_gray, sigma=0.05, seed=1)
    b = add_rician_noise(mid_gray, sigma=0.05, seed=2)
    assert not np.array_equal(a, b)


def test_zero_sigma_is_noop(mid_gray: np.ndarray) -> None:
    out = add_rician_noise(mid_gray, sigma=0.0, seed=0)
    assert np.array_equal(out, mid_gray)


def test_output_is_corrupted_at_nonzero_sigma(mid_gray: np.ndarray) -> None:
    out = add_rician_noise(mid_gray, sigma=0.08, seed=7)
    assert (out != mid_gray).any()


def test_output_stays_in_uint8_range(mid_gray: np.ndarray) -> None:
    out = add_rician_noise(mid_gray, sigma=0.15, seed=0)
    assert int(out.min()) >= 0
    assert int(out.max()) <= 255


def test_noise_floor_on_black_image() -> None:
    # Rician noise raises a zero-signal pixel above zero (noise floor).
    black = np.zeros((64, 64), dtype=np.uint8)
    out = add_rician_noise(black, sigma=0.10, seed=0)
    assert out.mean() > 0, "Rician noise floor should lift a zero image"


def test_negative_sigma_rejected(mid_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="sigma"):
        add_rician_noise(mid_gray, sigma=-0.01)


def test_non_array_rejected() -> None:
    with pytest.raises(TypeError):
        add_rician_noise([[128, 128], [128, 128]], sigma=0.05)  # type: ignore[arg-type]


def test_color_image_rejected() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="2-D"):
        add_rician_noise(rgb, sigma=0.05)


def test_generator_seed_is_threaded(mid_gray: np.ndarray) -> None:
    rng = np.random.default_rng(99)
    a = add_rician_noise(mid_gray, sigma=0.05, seed=rng)
    rng2 = np.random.default_rng(99)
    b = add_rician_noise(mid_gray, sigma=0.05, seed=rng2)
    assert np.array_equal(a, b)
