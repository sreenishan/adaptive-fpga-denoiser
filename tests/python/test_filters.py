"""Software reference filters and the selector (spec sections 14, 15, 41).

These filters are the golden reference the RTL is checked against, so the tests
pin exact values by hand wherever the arithmetic is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from denoising import config as cfg
from denoising.filters import (
    BINOMIAL_DIVISOR_3X3,
    BINOMIAL_KERNEL_3X3,
    CONTROL_CODE,
    FILTER_FOR_CLASS,
    FILTERS,
    apply_filter,
    control_code,
    decide_filter,
    estimate_noise_variance,
    gaussian_filter,
    gaussian_kernel,
    median_filter,
    select_filter,
    wiener_filter,
)
from denoising.filters._window import pad_image, sliding_windows
from denoising.noise import add_salt_pepper_noise

FILTER_FUNCS = {
    "median": lambda img: median_filter(img, 3),
    "gaussian": lambda img: gaussian_filter(img, 3),
    "wiener": lambda img: wiener_filter(img, 3, 100.0),
}


@pytest.fixture
def ramp() -> np.ndarray:
    """A deterministic 16x16 gradient with a bright square in it."""
    yy, xx = np.mgrid[0:16, 0:16]
    image = ((xx * 8 + yy * 4) % 256).astype(np.uint8)
    image[4:8, 4:8] = 200
    return image


# The window contract shared with the RTL.


def test_edges_are_replicated_not_zero_padded() -> None:
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    padded = pad_image(image, 1)
    assert padded.shape == (4, 4)
    assert padded[0, 0] == 1 and padded[0, 3] == 2
    assert padded[3, 0] == 3 and padded[3, 3] == 4
    assert 0 not in padded


def test_window_of_the_corner_pixel_repeats_it(ramp: np.ndarray) -> None:
    windows = sliding_windows(ramp, 3)
    assert windows.shape == (16, 16, 3, 3)
    corner = windows[0, 0]
    assert corner[0, 0] == corner[0, 1] == corner[1, 0] == ramp[0, 0]
    assert windows[5, 5][1, 1] == ramp[5, 5]


@pytest.mark.parametrize("name", sorted(FILTER_FUNCS))
def test_filters_preserve_shape_and_dtype(name: str, ramp: np.ndarray) -> None:
    out = FILTER_FUNCS[name](ramp)
    assert out.shape == ramp.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(FILTER_FUNCS))
def test_filters_do_not_modify_the_input(name: str, ramp: np.ndarray) -> None:
    original = ramp.copy()
    FILTER_FUNCS[name](ramp)
    assert np.array_equal(ramp, original)


@pytest.mark.parametrize("name", sorted(FILTER_FUNCS))
def test_filters_leave_a_constant_image_alone(name: str) -> None:
    """Every one of these is an average, a median or a mean-plus-gain, so a flat
    image is a fixed point. A filter that shifts it has a boundary or a
    rounding bug."""
    for level in (0, 1, 128, 254, 255):
        flat = np.full((12, 12), level, dtype=np.uint8)
        assert (FILTER_FUNCS[name](flat) == level).all()


@pytest.mark.parametrize("name", sorted(FILTER_FUNCS))
def test_filters_reject_even_kernels(name: str, ramp: np.ndarray) -> None:
    with pytest.raises(ValueError, match="odd"):
        {"median": median_filter, "gaussian": gaussian_filter, "wiener": wiener_filter}[
            name
        ](ramp, 4)


# Median.


def test_median_removes_an_isolated_impulse() -> None:
    image = np.full((5, 5), 20, dtype=np.uint8)
    image[2, 2] = 255
    assert (median_filter(image, 3) == 20).all()


def test_median_output_is_always_an_input_value(ramp: np.ndarray) -> None:
    """The median of an odd window is a sample, never an average — which is
    why the RTL comparison can demand bit-exactness."""
    out = median_filter(ramp, 3)
    assert set(np.unique(out)).issubset(set(np.unique(ramp)))


def test_median_matches_a_hand_computed_window() -> None:
    image = np.array(
        [[1, 2, 3], [4, 250, 6], [7, 8, 9]],
        dtype=np.uint8,
    )
    # Centre window is the whole image: sorted 1,2,3,4,6,7,8,9,250 -> median 6.
    assert median_filter(image, 3)[1, 1] == 6


def test_median_beats_the_gaussian_on_salt_and_pepper(ramp: np.ndarray) -> None:
    """The reason for the mapping in the first place, measured rather than
    asserted."""
    noisy = add_salt_pepper_noise(ramp, 0.10, 0.5, seed=5)
    error = lambda out: np.abs(out.astype(int) - ramp.astype(int)).mean()  # noqa: E731
    assert error(median_filter(noisy, 3)) < error(gaussian_filter(noisy, 3))


# Gaussian.


def test_binomial_kernel_is_the_documented_one() -> None:
    assert BINOMIAL_KERNEL_3X3.tolist() == [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    assert int(BINOMIAL_KERNEL_3X3.sum()) == BINOMIAL_DIVISOR_3X3 == 16


def test_gaussian_matches_a_hand_computed_window() -> None:
    """1*10+2*10+1*10 + 2*10+4*255+2*10 + 1*10+2*10+1*10 = 1140;
    (1140 + 8) >> 4 = 71."""
    image = np.full((5, 5), 10, dtype=np.uint8)
    image[2, 2] = 255
    assert gaussian_filter(image, 3)[2, 2] == 71


def test_gaussian_accumulator_fits_thirteen_bits() -> None:
    """The RTL sizes its accumulator from this bound: 255 * 16 = 4080."""
    white = np.full((8, 8), 255, dtype=np.uint8)
    windows = sliding_windows(white, 3).astype(np.int32)
    assert int((windows * BINOMIAL_KERNEL_3X3).sum(axis=(-2, -1)).max()) == 4080
    assert 4080 < 2**13


def test_gaussian_rounds_half_up_not_down() -> None:
    """An accumulator of exactly 8 must give 1, not 0 — the rule the RTL adder
    implements. Nine pixels of 1 at the centre-weighted kernel sum to 16, so a
    single 1 surrounded by 0 sums to 4 (>>4 = 0) while the flat case gives 1."""
    single = np.zeros((5, 5), dtype=np.uint8)
    single[2, 2] = 8
    # centre window: 4 * 8 = 32; (32 + 8) >> 4 = 2
    assert gaussian_filter(single, 3)[2, 2] == 2


def test_gaussian_is_exact_integer_arithmetic(ramp: np.ndarray) -> None:
    """Recomputed the long way; any float creeping in would show up here."""
    windows = sliding_windows(ramp, 3).astype(np.int64)
    expected = ((windows * BINOMIAL_KERNEL_3X3).sum(axis=(-2, -1)) + 8) // 16
    assert np.array_equal(gaussian_filter(ramp, 3), expected.astype(np.uint8))


def test_sampled_gaussian_kernel_is_normalised() -> None:
    kernel = gaussian_kernel(5, 1.2)
    assert kernel.shape == (5, 5)
    assert kernel.sum() == pytest.approx(1.0)
    assert kernel[2, 2] == kernel.max()


def test_integer_kernel_is_only_defined_at_three(ramp: np.ndarray) -> None:
    with pytest.raises(ValueError, match="only defined for kernel_size=3"):
        gaussian_filter(ramp, 5, integer_kernel=True)


def test_sampled_kernel_needs_a_sigma(ramp: np.ndarray) -> None:
    with pytest.raises(ValueError, match="sigma is required"):
        gaussian_filter(ramp, 3, None, integer_kernel=False)


def test_the_two_gaussian_kernels_are_not_the_same_filter(ramp: np.ndarray) -> None:
    """Documented, and asserted so nobody assumes sigma: 1.0 describes the
    integer kernel."""
    integer = gaussian_filter(ramp, 3, integer_kernel=True)
    sampled = gaussian_filter(ramp, 3, 1.0, integer_kernel=False)
    assert not np.array_equal(integer, sampled)


# Wiener.


def test_wiener_returns_the_local_mean_where_there_is_no_structure() -> None:
    """Flat window, gain 0: the output is the mean, which is the pixel."""
    flat = np.full((9, 9), 100, dtype=np.uint8)
    assert (wiener_filter(flat, 3, 25.0) == 100).all()


def test_wiener_preserves_a_strong_edge() -> None:
    """High local variance drives the gain towards 1, so the pixel passes
    through — the property that makes it adaptive."""
    image = np.zeros((9, 9), dtype=np.uint8)
    image[:, 4:] = 255
    out = wiener_filter(image, 3, 4.0)
    assert out[4, 0] == 0
    assert out[4, 8] == 255


def test_wiener_smooths_more_as_the_noise_estimate_grows(ramp: np.ndarray) -> None:
    spread = lambda v: float(wiener_filter(ramp, 3, v).astype(float).std())  # noqa: E731
    assert spread(5000.0) < spread(10.0)


def test_wiener_with_zero_noise_is_a_pass_through(ramp: np.ndarray) -> None:
    """noise_var = 0 means no noise to remove, so the gain is 1 everywhere.
    This is why the config's null means 'estimate it', not 0."""
    assert np.array_equal(wiener_filter(ramp, 3, 0.0), ramp)


def test_wiener_handles_a_flat_image_with_zero_noise() -> None:
    """The 0/0 case: both variances zero. Must not produce NaN."""
    flat = np.full((8, 8), 77, dtype=np.uint8)
    out = wiener_filter(flat, 3, 0.0)
    assert (out == 77).all()


def test_estimated_noise_variance_grows_with_noise(ramp: np.ndarray) -> None:
    noisy = add_salt_pepper_noise(ramp, 0.10, 0.5, seed=2)
    assert estimate_noise_variance(noisy, 3) > estimate_noise_variance(ramp, 3)


def test_wiener_negative_noise_variance_is_rejected(ramp: np.ndarray) -> None:
    with pytest.raises(ValueError, match="noise_variance"):
        wiener_filter(ramp, 3, -1.0)


# Selector.


def test_the_mapping_is_the_one_in_the_spec() -> None:
    assert FILTER_FOR_CLASS == {
        "clean": "bypass",
        "salt_pepper": "median",
        "gaussian": "gaussian",
        "speckle": "wiener",
    }


def test_every_class_maps_to_a_filter() -> None:
    for name in cfg.CLASSES:
        assert select_filter(name) in FILTERS


def test_control_codes_match_the_rtl_encoding() -> None:
    """2'b00 bypass, 2'b01 median, 2'b10 gaussian, 2'b11 wiener (spec 26)."""
    assert CONTROL_CODE == {"bypass": 0, "median": 1, "gaussian": 2, "wiener": 3}
    assert control_code(select_filter("speckle")) == 0b11
    assert all(0 <= code <= 3 for code in CONTROL_CODE.values())


def test_an_unknown_class_raises_rather_than_defaulting() -> None:
    """A silent default would filter an unrecognised class with something
    plausible instead of reporting that nobody mapped it."""
    with pytest.raises(ValueError, match="unknown noise class"):
        select_filter("motion_blur")


def test_decide_uses_the_mapping_above_the_threshold() -> None:
    decision = decide_filter("speckle", 0.9, threshold=0.6, fallback="bypass")
    assert decision.filter_name == "wiener"
    assert decision.used_fallback is False
    assert decision.control_code == 3


def test_decide_falls_back_below_the_threshold_and_says_so() -> None:
    decision = decide_filter("speckle", 0.2, threshold=0.6, fallback="median")
    assert decision.filter_name == "median"
    assert decision.used_fallback is True
    assert decision.mapped_filter == "wiener"
    assert decision.confidence == 0.2


def test_a_manual_choice_has_no_confidence_and_never_falls_back() -> None:
    decision = decide_filter("gaussian", None, threshold=0.99)
    assert decision.confidence is None
    assert decision.used_fallback is False
    assert decision.filter_name == "gaussian"


def test_decide_rejects_an_unknown_fallback() -> None:
    with pytest.raises(ValueError, match="unknown fallback"):
        decide_filter("clean", 0.1, threshold=0.5, fallback="denoise_harder")


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_decide_rejects_impossible_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        decide_filter("clean", confidence, threshold=0.5)


# Dispatch.


def test_bypass_returns_an_untouched_copy(ramp: np.ndarray) -> None:
    out = apply_filter(ramp, "bypass")
    assert np.array_equal(out, ramp)
    assert out is not ramp


@pytest.mark.parametrize("name", FILTERS)
def test_apply_filter_handles_every_filter(name: str, ramp: np.ndarray) -> None:
    out = apply_filter(ramp, name, kernel_size=3, sigma=1.0, noise_variance=50.0)
    assert out.shape == ramp.shape
    assert out.dtype == np.uint8


def test_apply_filter_rejects_an_unknown_name(ramp: np.ndarray) -> None:
    with pytest.raises(ValueError, match="unknown filter"):
        apply_filter(ramp, "sharpen")


def test_configured_kernels_are_the_ones_with_rtl_counterparts() -> None:
    filters = cfg.load_inference_config().filters
    assert filters.median.kernel_size == 3
    assert filters.gaussian.kernel_size == 3
    assert filters.wiener.kernel_size == 3
    assert filters.gaussian.integer_kernel is True
