"""Noise generators (spec sections 7 and 41).

Every test uses an explicit seed or an explicit constant image; none depends on
wall-clock entropy except the two that deliberately check ``seed=None`` is not
reproducible.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from denoising import config as cfg
from denoising.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
)

Generator = Callable[[np.ndarray, object], np.ndarray]

#: Each generator wrapped to a common (image, seed) signature, at an intensity
#: strong enough that "did anything happen?" is unambiguous.
GENERATORS: list[tuple[str, Generator]] = [
    ("salt_pepper", lambda img, seed: add_salt_pepper_noise(img, 0.05, 0.5, seed)),
    ("gaussian", lambda img, seed: add_gaussian_noise(img, 0.0, 0.08, seed)),
    ("speckle", lambda img, seed: add_speckle_noise(img, 0.08, seed)),
]
_IDS = [name for name, _ in GENERATORS]
_FUNCS = [func for _, func in GENERATORS]


@pytest.fixture
def mid_gray() -> np.ndarray:
    """A 64x64 image of 128. Mid-grey so any corruption is a visible change."""
    return np.full((64, 64), 128, dtype=np.uint8)


# The contract every generator shares.


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_shape_and_dtype_are_preserved(generate: Generator, mid_gray: np.ndarray) -> None:
    out = generate(mid_gray, 7)
    assert out.shape == mid_gray.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_input_is_not_modified(generate: Generator, mid_gray: np.ndarray) -> None:
    original = mid_gray.copy()
    generate(mid_gray, 7)
    assert np.array_equal(mid_gray, original)


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_the_same_seed_gives_the_same_image(generate: Generator, mid_gray: np.ndarray) -> None:
    assert np.array_equal(generate(mid_gray, 12345), generate(mid_gray, 12345))


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_different_seeds_give_different_images(generate: Generator, mid_gray: np.ndarray) -> None:
    assert not np.array_equal(generate(mid_gray, 1), generate(mid_gray, 2))


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_no_seed_is_not_reproducible(generate: Generator, mid_gray: np.ndarray) -> None:
    """seed=None draws fresh entropy, which is a documented choice, not a bug.
    Over 4096 pixels a collision is not a thing that happens."""
    assert not np.array_equal(generate(mid_gray, None), generate(mid_gray, None))


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_a_generator_object_is_equivalent_to_its_seed(
    generate: Generator, mid_gray: np.ndarray
) -> None:
    assert np.array_equal(generate(mid_gray, np.random.default_rng(99)), generate(mid_gray, 99))


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_a_shared_generator_advances_between_calls(
    generate: Generator, mid_gray: np.ndarray
) -> None:
    """Threading one stream through a dataset build must not repeat itself."""
    rng = np.random.default_rng(4)
    assert not np.array_equal(generate(mid_gray, rng), generate(mid_gray, rng))


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_output_stays_inside_the_pixel_range(generate: Generator) -> None:
    """uint8 arithmetic wraps round silently; clipping must happen first."""
    for level in (0, 1, 128, 254, 255):
        out = generate(np.full((32, 32), level, dtype=np.uint8), 3)
        assert out.min() >= 0
        assert out.max() <= 255


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_non_array_input_is_rejected(generate: Generator) -> None:
    with pytest.raises(TypeError, match="numpy array"):
        generate([[1, 2], [3, 4]], 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_non_uint8_input_is_rejected(generate: Generator) -> None:
    with pytest.raises(ValueError, match="uint8"):
        generate(np.zeros((8, 8), dtype=np.float32), 0)


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_colour_input_is_rejected(generate: Generator) -> None:
    """Colour is out of scope; silently noising one channel layout or another
    would be a guess about what the caller meant."""
    with pytest.raises(ValueError, match="2-D grayscale"):
        generate(np.zeros((8, 8, 3), dtype=np.uint8), 0)


@pytest.mark.parametrize("generate", _FUNCS, ids=_IDS)
def test_empty_input_is_rejected(generate: Generator) -> None:
    with pytest.raises(ValueError, match="empty"):
        generate(np.zeros((0, 0), dtype=np.uint8), 0)


# Salt and pepper.


def test_salt_pepper_zero_amount_is_a_no_op(mid_gray: np.ndarray) -> None:
    assert np.array_equal(add_salt_pepper_noise(mid_gray, 0.0, seed=1), mid_gray)


def test_salt_pepper_full_amount_corrupts_everything(mid_gray: np.ndarray) -> None:
    out = add_salt_pepper_noise(mid_gray, 1.0, seed=1)
    assert np.isin(out, (0, 255)).all()


@pytest.mark.parametrize("amount", [0.02, 0.05, 0.10, 0.25])
def test_salt_pepper_corrupts_exactly_the_requested_fraction(
    amount: float, mid_gray: np.ndarray
) -> None:
    """A fixed count, not a per-pixel coin flip: the label on a sample should
    describe that sample, not the distribution it came from."""
    out = add_salt_pepper_noise(mid_gray, amount, seed=11)
    expected = round(amount * mid_gray.size)
    assert int((out != mid_gray).sum()) == expected


def test_salt_only(mid_gray: np.ndarray) -> None:
    out = add_salt_pepper_noise(mid_gray, 0.10, salt_vs_pepper=1.0, seed=5)
    assert (out == 255).sum() == round(0.10 * mid_gray.size)
    assert (out == 0).sum() == 0


def test_pepper_only(mid_gray: np.ndarray) -> None:
    out = add_salt_pepper_noise(mid_gray, 0.10, salt_vs_pepper=0.0, seed=5)
    assert (out == 0).sum() == round(0.10 * mid_gray.size)
    assert (out == 255).sum() == 0


def test_salt_and_pepper_are_balanced_at_one_half(mid_gray: np.ndarray) -> None:
    """An odd corruption count cannot split evenly; it may land either side of
    the half, but the two halves must still account for every pixel."""
    corrupted = round(0.20 * mid_gray.size)
    out = add_salt_pepper_noise(mid_gray, 0.20, salt_vs_pepper=0.5, seed=5)
    salt = int((out == 255).sum())
    pepper = int((out == 0).sum())
    assert salt + pepper == corrupted
    assert abs(salt - pepper) <= 1


@pytest.mark.parametrize("amount", [-0.01, 1.01])
def test_salt_pepper_amount_out_of_range_is_rejected(amount: float, mid_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="amount"):
        add_salt_pepper_noise(mid_gray, amount, seed=1)


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_salt_vs_pepper_out_of_range_is_rejected(ratio: float, mid_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="salt_vs_pepper"):
        add_salt_pepper_noise(mid_gray, 0.1, ratio, seed=1)


# Gaussian.


def test_gaussian_with_no_noise_is_a_no_op(mid_gray: np.ndarray) -> None:
    assert np.array_equal(add_gaussian_noise(mid_gray, 0.0, 0.0, seed=1), mid_gray)


def test_gaussian_mean_shifts_every_pixel_equally(mid_gray: np.ndarray) -> None:
    """0.1 of full scale on a 128 image: 128/255 + 0.1 = 0.60196, x255 = 153.5,
    which rounds half-to-even to 154."""
    out = add_gaussian_noise(mid_gray, mean=0.1, sigma=0.0, seed=1)
    assert out.min() == out.max() == 154


def test_gaussian_sigma_matches_the_requested_spread() -> None:
    """Measured on mid-grey where clipping is negligible at this sigma."""
    image = np.full((256, 256), 128, dtype=np.uint8)
    out = add_gaussian_noise(image, 0.0, 0.05, seed=42).astype(np.float64)
    assert out.std() == pytest.approx(0.05 * 255, rel=0.05)
    assert out.mean() == pytest.approx(128, abs=1.0)


def test_gaussian_saturates_instead_of_wrapping() -> None:
    white = np.full((32, 32), 255, dtype=np.uint8)
    black = np.zeros((32, 32), dtype=np.uint8)
    assert (add_gaussian_noise(white, mean=0.5, sigma=0.0, seed=1) == 255).all()
    assert (add_gaussian_noise(black, mean=-0.5, sigma=0.0, seed=1) == 0).all()


def test_gaussian_negative_sigma_is_rejected(mid_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="sigma"):
        add_gaussian_noise(mid_gray, 0.0, -0.1, seed=1)


@pytest.mark.parametrize("mean", [float("nan"), float("inf")])
def test_gaussian_non_finite_mean_is_rejected(mean: float, mid_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="mean"):
        add_gaussian_noise(mid_gray, mean, 0.05, seed=1)


# Speckle.


def test_speckle_with_zero_variance_is_a_no_op(mid_gray: np.ndarray) -> None:
    assert np.array_equal(add_speckle_noise(mid_gray, 0.0, seed=1), mid_gray)


def test_speckle_leaves_black_untouched() -> None:
    """The defining property of a multiplicative model: 0 * anything is 0.
    If this ever fails, the model has quietly become additive."""
    black = np.zeros((64, 64), dtype=np.uint8)
    assert (add_speckle_noise(black, 0.10, seed=1) == 0).all()


def test_speckle_scales_with_pixel_intensity() -> None:
    dark = np.full((256, 256), 40, dtype=np.uint8)
    bright = np.full((256, 256), 160, dtype=np.uint8)
    dark_std = add_speckle_noise(dark, 0.01, seed=8).astype(np.float64).std()
    bright_std = add_speckle_noise(bright, 0.01, seed=8).astype(np.float64).std()
    assert bright_std / dark_std == pytest.approx(160 / 40, rel=0.15)


def test_speckle_argument_is_a_variance_not_a_standard_deviation() -> None:
    """variance=0.04 on a 100 image gives std = sqrt(0.04) * 100 = 20.
    Passing the argument straight through as a std would give 4."""
    image = np.full((256, 256), 100, dtype=np.uint8)
    out = add_speckle_noise(image, 0.04, seed=13).astype(np.float64)
    assert out.std() == pytest.approx(20.0, rel=0.10)


def test_speckle_negative_variance_is_rejected(mid_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="variance"):
        add_speckle_noise(mid_gray, -0.01, seed=1)


# The configured intensities must actually do something.


def test_every_configured_intensity_produces_visible_noise() -> None:
    """A configured level too weak to change any pixel would put mislabelled
    clean images into three of the four classes."""
    dataset = cfg.load_dataset_config()
    image = np.full((64, 64), 128, dtype=np.uint8)
    noise = dataset.noise

    for amount in noise.salt_pepper.amounts:
        out = add_salt_pepper_noise(image, amount, noise.salt_pepper.salt_vs_pepper, seed=1)
        assert (out != image).any(), f"salt_pepper amount={amount} changed nothing"

    for sigma in noise.gaussian.sigmas:
        out = add_gaussian_noise(image, noise.gaussian.mean, sigma, seed=1)
        assert (out != image).any(), f"gaussian sigma={sigma} changed nothing"

    for variance in noise.speckle.variances:
        out = add_speckle_noise(image, variance, seed=1)
        assert (out != image).any(), f"speckle variance={variance} changed nothing"
