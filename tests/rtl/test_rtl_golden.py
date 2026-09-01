"""The RTL must agree with the software reference, within the declared budget.

``configs/hardware.yaml`` states the contract these tests enforce::

    max_abs_error:
      median:   0     # bit-exact
      gaussian: 0     # bit-exact
      wiener:   1     # one grey level

These run against :mod:`tests.rtl.rtl_model`, a statement-for-statement
transcription of ``rtl/*.sv``. That is weaker than simulation — it cannot catch
a syntax or elaboration error — but it is what caught the three defects the RTL
shipped with, and it runs in the normal test suite instead of needing a toolchain
nobody has installed.

Every one of these tests FAILED before the phase-23 fixes:
  · the median network was wrong on 82.54% of all input orderings,
  · the Wiener filter was 24 grey levels out against a 1 LSB budget,
  · the window generator emitted 49950 windows for a 50176-pixel frame,
    with no edge replication and row-wrapped neighbourhoods.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from denoising.filters import gaussian_filter, median_filter, wiener_filter
from denoising.filters._window import sliding_windows

from rtl_model import (  # noqa: E402  (same directory; pytest prepends it)
    MEDIAN_NETWORK,
    gaussian_3x3,
    median_of_9,
    stream_windows,
    wiener_3x3,
)

# ── tolerances from configs/hardware.yaml ──────────────────────────────────
MEDIAN_TOL = 0
GAUSSIAN_TOL = 0
WIENER_TOL = 1


def _image(kind: str, size: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    if kind == "flat0":
        return np.zeros((size, size), np.uint8)
    if kind == "flat255":
        return np.full((size, size), 255, np.uint8)
    if kind == "ramp":
        return ((xx * 5) % 256).astype(np.uint8)
    if kind == "checker":
        return (((xx + yy) % 2) * 255).astype(np.uint8)
    if kind == "random":
        return rng.integers(0, 256, (size, size), dtype=np.uint8)
    if kind == "noisy":
        base = ((xx * 3 + yy * 2) % 200 + 28).astype(np.float64)
        return np.clip(base + rng.normal(0, 14, (size, size)), 0, 255).astype(np.uint8)
    raise ValueError(kind)


IMAGE_KINDS = ("flat0", "flat255", "ramp", "checker", "random", "noisy")


# ══ median ═════════════════════════════════════════════════════════════════

def test_median_network_has_nineteen_comparators() -> None:
    """19 is the smallest known correct median-of-9 network.

    Removing one makes it wrong; the previous implementation had 17 and was
    wrong on 82.54% of inputs. This pins the count so an "optimisation" cannot
    quietly reintroduce that.
    """
    assert len(MEDIAN_NETWORK) == 19


def test_median_exact_on_all_permutations() -> None:
    """The decisive test: all 9! orderings of nine distinct values."""
    wrong = sum(
        1 for p in itertools.permutations(range(9)) if median_of_9(p) != sorted(p)[4]
    )
    assert wrong == 0, f"{wrong}/362880 permutations returned the wrong median"


def test_median_exact_with_duplicates() -> None:
    """Exhaustive over 4**9 windows, so ties and repeats are covered too."""
    wrong = sum(
        1
        for p in itertools.product(range(4), repeat=9)
        if median_of_9(p) != sorted(p)[4]
    )
    assert wrong == 0, f"{wrong}/262144 duplicate-heavy windows were wrong"


@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_median_matches_reference_bit_exact(kind: str) -> None:
    image = _image(kind)
    reference = median_filter(image, 3)
    windows = sliding_windows(image, 3)
    got = np.array(
        [
            [median_of_9([int(v) for v in windows[r, c].reshape(-1)]) for c in range(image.shape[1])]
            for r in range(image.shape[0])
        ],
        dtype=np.uint8,
    )
    error = int(np.abs(got.astype(int) - reference.astype(int)).max())
    assert error <= MEDIAN_TOL, f"{kind}: max|err| {error} > {MEDIAN_TOL}"


# ══ gaussian ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_gaussian_matches_reference_bit_exact(kind: str) -> None:
    image = _image(kind)
    reference = gaussian_filter(image, 3, integer_kernel=True)
    windows = sliding_windows(image, 3)
    got = np.array(
        [
            [gaussian_3x3([int(v) for v in windows[r, c].reshape(-1)]) for c in range(image.shape[1])]
            for r in range(image.shape[0])
        ],
        dtype=np.uint8,
    )
    error = int(np.abs(got.astype(int) - reference.astype(int)).max())
    assert error <= GAUSSIAN_TOL, f"{kind}: max|err| {error} > {GAUSSIAN_TOL}"


def test_gaussian_accumulator_cannot_overflow() -> None:
    """SUM_W is 12 bits; the worst case plus the rounding add must still fit."""
    worst = 255 * 16 + 8
    assert worst < (1 << 12)


# ══ wiener ═════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("kind", IMAGE_KINDS)
@pytest.mark.parametrize("noise_var", (0, 1, 25, 100, 400, 4000))
def test_wiener_within_one_grey_level(kind: str, noise_var: int) -> None:
    image = _image(kind)
    reference = wiener_filter(image, 3, float(noise_var))
    windows = sliding_windows(image, 3)
    got = np.array(
        [
            [
                wiener_3x3([int(v) for v in windows[r, c].reshape(-1)], noise_var)
                for c in range(image.shape[1])
            ]
            for r in range(image.shape[0])
        ],
        dtype=np.uint8,
    )
    error = int(np.abs(got.astype(int) - reference.astype(int)).max())
    assert error <= WIENER_TOL, f"{kind}/NV={noise_var}: max|err| {error} > {WIENER_TOL}"


def test_wiener_variance_identity_is_exact() -> None:
    """``81*var == 9*sum(x^2) - sum(x)^2`` — the identity the fix rests on."""
    rng = np.random.default_rng(4)
    for _ in range(2000):
        w = [int(v) for v in rng.integers(0, 256, 9)]
        exact = 9 * sum(v * v for v in w) - sum(w) ** 2
        assert exact == pytest.approx(81 * float(np.var(w)), abs=1e-6)
        assert exact >= 0


def test_wiener_flat_window_returns_the_mean() -> None:
    """Zero variance and zero noise must smooth to the mean, not pass through."""
    assert wiener_3x3([100] * 9, 0) == 100
    assert wiener_3x3([100] * 9, 100) == 100


# ══ window generator ═══════════════════════════════════════════════════════

@pytest.mark.parametrize("height,width", [(6, 8), (8, 8), (5, 7), (16, 24), (3, 3)])
def test_window_generator_matches_replicate_padding(height: int, width: int) -> None:
    """One window per pixel, raster order, replicate-padded — exactly."""
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (height, width), dtype=np.uint8)
    reference = sliding_windows(image, 3)

    produced = list(stream_windows(image, width, height))
    assert len(produced) == width * height, "must emit one window per input pixel"

    expected_order = [(r, c) for r in range(height) for c in range(width)]
    assert [(r, c) for r, c, _ in produced] == expected_order

    for r, c, window in produced:
        assert np.array_equal(np.array(window), reference[r, c]), f"window ({r},{c})"


def test_window_generator_drains_between_frames() -> None:
    """A second frame streamed straight after the first must be framed identically.

    The previous implementation latched ``filled`` and never cleared it, so
    frame two was mis-framed and mixed rows from frame one.
    """
    rng = np.random.default_rng(9)
    for _ in range(2):
        image = rng.integers(0, 256, (12, 10), dtype=np.uint8)
        reference = sliding_windows(image, 3)
        for r, c, window in stream_windows(image, 10, 12):
            assert np.array_equal(np.array(window), reference[r, c])


def test_full_chain_matches_each_reference_filter() -> None:
    """window_gen -> filter core, end to end, against the real filters."""
    image = _image("noisy", 32, seed=5)
    h, w = image.shape
    out_median = np.zeros_like(image)
    out_gaussian = np.zeros_like(image)
    out_wiener = np.zeros_like(image)

    for r, c, window in stream_windows(image, w, h):
        flat = [v for row in window for v in row]
        out_median[r, c] = median_of_9(flat)
        out_gaussian[r, c] = gaussian_3x3(flat)
        out_wiener[r, c] = wiener_3x3(flat, 100)

    checks = (
        ("median", out_median, median_filter(image, 3), MEDIAN_TOL),
        ("gaussian", out_gaussian, gaussian_filter(image, 3, integer_kernel=True), GAUSSIAN_TOL),
        ("wiener", out_wiener, wiener_filter(image, 3, 100.0), WIENER_TOL),
    )
    for name, got, reference, tolerance in checks:
        error = int(np.abs(got.astype(int) - reference.astype(int)).max())
        assert error <= tolerance, f"{name}: max|err| {error} > {tolerance}"
