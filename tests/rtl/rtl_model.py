"""Bit-accurate Python models of the RTL in ``rtl/``.

Why this file exists
--------------------
No simulator is installed, and the SystemVerilog had never been executed by
anything when three of its five modules were found to be wrong. A model that
mirrors the RTL *statement for statement* is not a substitute for simulation —
it cannot catch a syntax error, an elaboration failure or a timing violation —
but it does catch the defect class that actually bit this project: fixed-point
slice errors, comparator networks that do not compute what their comment says,
and boundary handling that was never implemented.

Every function here is a transcription of the corresponding ``.sv`` file **as
written**, not as intended. When the RTL changes, change this to match and
re-run :mod:`tests.rtl.test_rtl_golden`. If the two ever drift, the tests stop
testing the hardware and start testing themselves.

Tolerances come from ``configs/hardware.yaml``: median 0, gaussian 0, wiener 1.
"""

from __future__ import annotations

from typing import Sequence

__all__ = [
    "MEDIAN_NETWORK",
    "median_of_9",
    "gaussian_3x3",
    "wiener_3x3",
    "stream_windows",
]

#: The 19 compare-exchange steps of ``rtl/median_filter.sv``, in file order.
#: After ``CS(i, j)`` the smaller value is at index ``i``. The descending pairs
#: (4, 2) and (6, 4) are deliberate — the minimum lands in the first index named.
MEDIAN_NETWORK: tuple[tuple[int, int], ...] = (
    (1, 2), (4, 5), (7, 8),
    (0, 1), (3, 4), (6, 7),
    (1, 2), (4, 5), (7, 8),
    (0, 3), (5, 8), (4, 7),
    (3, 6), (1, 4), (2, 5),
    (4, 7), (4, 2), (6, 4), (4, 2),
)


def median_of_9(window: Sequence[int]) -> int:
    """``rtl/median_filter.sv`` — the 19-comparator median-of-9 network.

    Args:
        window: Nine pixel values, row-major.

    Returns:
        The median. Verified equal to ``sorted(window)[4]`` over all 9!
        permutations of distinct values and the exhaustive ``4**9`` space with
        duplicates.
    """
    v = list(window)
    for a, b in MEDIAN_NETWORK:
        if v[a] > v[b]:
            v[a], v[b] = v[b], v[a]
    return v[4]


def gaussian_3x3(window: Sequence[int]) -> int:
    """``rtl/gaussian_filter.sv`` — binomial kernel, round-half-up, >>4."""
    weights = (1, 2, 1, 2, 4, 2, 1, 2, 1)
    return (sum(w * p for w, p in zip(weights, window)) + 8) >> 4


def wiener_3x3(window: Sequence[int], noise_var: int) -> int:
    """``rtl/wiener_filter.sv`` — exact-integer local-statistics Wiener filter.

    ``81 * variance = 9 * sum(x**2) - sum(x)**2`` is exact in integers, so the
    local mean is never rounded before being squared. That cancellation is what
    made the previous formulation 19 grey levels wrong against a 1 LSB budget.

    Args:
        window: Nine pixel values, row-major; index 4 is the centre.
        noise_var: Noise power in squared grey levels (the RTL's ``NOISE_VAR``).
    """
    s = sum(window)
    s2 = sum(p * p for p in window)
    v81 = 9 * s2 - s * s                     # = 81 * variance, exact, >= 0
    nv81 = 81 * noise_var

    numerator = v81 - nv81 if v81 > nv81 else 0
    denominator = v81 if v81 > nv81 else nv81

    if denominator == 0:
        # Flat window and zero noise: nothing to attenuate. The reference floors
        # the denominator at epsilon, so the gain goes to zero and the output is
        # the local mean.
        gain_q8 = 0
    else:
        gain_q8 = min(255, (numerator << 8) // denominator)

    acc = (s << 8) + gain_q8 * (9 * window[4] - s) + 1152   # +9<<7, round half up
    if acc <= 0:
        return 0
    quotient = acc // 2304                                  # 9 << 8
    return 255 if quotient > 255 else quotient


def stream_windows(image, width: int, height: int):
    """``rtl/window_gen.sv`` + ``rtl/line_buffer.sv`` — cycle-accurate.

    Models the host protocol: ``we`` high for ``width * height`` pixels, then
    ``flush`` high for ``width + 1`` cycles to drain.

    Yields:
        ``(row, col, window)`` for every centre pixel in raster order, where
        *window* is a 3x3 list of lists with replicate padding. Exactly
        ``width * height`` windows are produced.
    """
    sr0 = [0] * width
    sr1 = [0] * width
    col = [[0] * 3 for _ in range(3)]
    pixels = [int(v) for row in image for v in row]
    total = width * height

    for t in range(total + width + 1):
        flushing = t >= total
        live = sr0[width - 1] if flushing else pixels[t]

        # Taps are read BEFORE the shift, giving delays of 0, width and 2*width.
        taps = (live, sr0[width - 1], sr1[width - 1])
        col = [[col[r][1], col[r][2], taps[r]] for r in range(3)]
        sr0, sr1 = [live] + sr0[: width - 1], [sr0[width - 1]] + sr1[: width - 1]

        centre = t - width - 1
        if not 0 <= centre < total:
            continue
        cr, cc = divmod(centre, width)

        at_left, at_right = cc == 0, cc == width - 1

        def clamped(r: int) -> list[int]:
            left, mid, right = col[r]
            return [mid if at_left else left, mid, mid if at_right else right]

        # col[0] is the BOTTOM row (the live pixel trails the centre), col[1]
        # the centre row, col[2] the top.
        top, middle, bottom = clamped(2), clamped(1), clamped(0)
        if cr == 0:
            top = middle
        if cr == height - 1:
            bottom = middle

        yield cr, cc, [top, middle, bottom]
