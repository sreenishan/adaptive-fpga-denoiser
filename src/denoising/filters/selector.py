"""Automatic filter selection (spec sections 15 and 34).

The mapping from noise class to filter lives **here and nowhere else**. It is
read by the Python pipeline and it defines the 2-bit code sent to
``rtl/filter_controller.sv``; a second copy somewhere else is how the
software and the hardware come to disagree about what "speckle" means while
both look correct in isolation.

```text
clean       -> bypass    (2'b00)
salt_pepper -> median    (2'b01)
gaussian    -> gaussian  (2'b10)
speckle     -> wiener    (2'b11)
```

With severity enabled, the class picks a *row* and the estimated severity
picks the step — a filter and a number of passes — from :data:`SEVERITY_POLICY`.
Every entry was chosen by measurement, not intuition: each is the strategy with
the best mean PSNR over 40 synthetic sources at that class and level (a 0.1 dB
margin counts as a tie, which goes to fewer passes, then to the class default).
The figures are recorded beside the table.

Low confidence is handled here too, and it is handled by *saying so*: a
prediction below the configured threshold routes to the configured fallback and
the decision records that it did. Nothing about a low-confidence classification
is hidden, per spec section 34.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from ..config import CLASSES

__all__ = [
    "ALL_FILTERS",
    "FILTERS",
    "FilterStep",
    "SEVERITY_LEVELS",
    "SEVERITY_POLICY",
    "SOFTWARE_FILTERS",
    "FILTER_FOR_CLASS",
    "CONTROL_CODE",
    "FilterDecision",
    "select_filter",
    "decide_filter",
    "control_code",
]

#: Every filter the system can apply, in control-code order.
FILTERS: Final[tuple[str, ...]] = ("bypass", "median", "gaussian", "wiener")

#: The mapping. One definition, used by software and hardware alike.
FILTER_FOR_CLASS: Final[Mapping[str, str]] = {
    "clean": "bypass",
    "salt_pepper": "median",
    "gaussian": "gaussian",
    "speckle": "wiener",
}

#: The 2-bit code for each filter, as driven into the RTL controller.
CONTROL_CODE: Final[Mapping[str, int]] = {name: index for index, name in enumerate(FILTERS)}

# A class with no filter would fall through to a default at runtime, on some
# image, in front of somebody. Checked at import instead.
assert set(FILTER_FOR_CLASS) == set(CLASSES), "every class must map to a filter"
assert set(FILTER_FOR_CLASS.values()) <= set(FILTERS), "unknown filter in the mapping"

#: Filters with a software implementation and NO RTL counterpart. Kept out of
#: :data:`FILTERS` on purpose: that tuple is the hardware's control-code space,
#: and a filter in it would be assumed to run on the FPGA.
SOFTWARE_FILTERS: Final[tuple[str, ...]] = ("adaptive_median",)

#: Everything :func:`~denoising.filters.apply_filter` can run.
ALL_FILTERS: Final[tuple[str, ...]] = FILTERS + SOFTWARE_FILTERS

#: Severity levels, least to most severe.
SEVERITY_LEVELS: Final[tuple[str, ...]] = ("low", "medium", "high")


@dataclass(frozen=True)
class FilterStep:
    """One filtering decision: which filter, and how many passes of it."""

    filter_name: str
    passes: int = 1


#: (class, severity) -> step. ``clean`` has no row: an image with no noise has
#: no severity, and bypass needs no strength. Mean PSNR (dB) behind each entry,
#: 40 synthetic sources per cell, 3x3 kernels throughout:
#:
#:   salt_pepper  2%  median x1 49.61 | x2 49.53 | adaptive 45.87  -> median x1 (tie, fewer passes)
#:                5%  median x1 45.49 | x2 46.03 | adaptive 44.64  -> median x2
#:               10%  median x1 40.76 | x2 43.39 | adaptive 43.55  -> adaptive median
#:               30%  median x1 24.52 | x2 33.58 | adaptive 39.04  (why High is adaptive)
#:   gaussian  0.03  gauss x1 36.81 | x2 37.31 | wiener x1 38.55 | x2 41.35  -> wiener x2
#:             0.06  gauss x1 32.26 | x2 33.99 | wiener x1 32.02 | x2 34.84  -> wiener x2
#:             0.10  gauss x1 28.28 | x2 30.55 | wiener x1 27.47 | x2 30.09  -> gaussian x2
#:   speckle   0.03  wiener x1 26.97 | x2 29.18 | gauss x1 29.17               -> wiener x2
#:             0.06  wiener x1 24.15 | x2 26.32 | gauss x1 26.40  (tie)        -> wiener x2
#:             0.10  wiener x1 22.16 | x2 24.31 | gauss x1 24.32  (tie)        -> wiener x2
#:
#: Speckle ties go to wiener, the class default: a single Gaussian pass matches
#: two Wiener passes within 0.08 dB and would halve the hardware passes, but
#: that is a design decision to make on purpose, not to slip in through a tie.
SEVERITY_POLICY: Final[Mapping[str, Mapping[str, FilterStep]]] = {
    "salt_pepper": {
        "low": FilterStep("median", 1),
        "medium": FilterStep("median", 2),
        "high": FilterStep("adaptive_median", 1),
    },
    "gaussian": {
        "low": FilterStep("wiener", 2),
        "medium": FilterStep("wiener", 2),
        "high": FilterStep("gaussian", 2),
    },
    "speckle": {
        "low": FilterStep("wiener", 2),
        "medium": FilterStep("wiener", 2),
        "high": FilterStep("wiener", 2),
    },
}

assert set(SEVERITY_POLICY) == set(CLASSES) - {"clean"}, "every noisy class needs a policy row"
for _row in SEVERITY_POLICY.values():
    assert set(_row) == set(SEVERITY_LEVELS), "every row needs every severity level"
    for _step in _row.values():
        assert _step.filter_name in ALL_FILTERS, f"unknown filter {_step.filter_name!r} in policy"
        assert _step.passes >= 1, "a step must run at least once"


@dataclass(frozen=True)
class FilterDecision:
    """What was chosen, and whether the classifier's answer was used.

    Attributes:
        filter_name: The filter to apply, one of :data:`ALL_FILTERS`.
        noise_class: The predicted class.
        confidence: The classifier's confidence, or ``None`` when the class was
            supplied by hand rather than predicted.
        used_fallback: True when confidence was below the threshold and
            *filter_name* is the fallback rather than the mapped filter.
        mapped_filter: What the mapping would have chosen. Kept even when the
            fallback wins, so the report can say what was overridden.
        threshold: The threshold the confidence was compared against.
        passes: How many times *filter_name* is applied in sequence.
        severity: The severity level that chose the step, or ``None`` when
            severity was not used (disabled, clean class, or the fallback won).
    """

    filter_name: str
    noise_class: str
    confidence: float | None
    used_fallback: bool
    mapped_filter: str
    threshold: float | None
    passes: int = 1
    severity: str | None = None

    @property
    def control_code(self) -> int | None:
        """The 2-bit code for the RTL controller, or ``None`` for a software-only filter.

        ``None`` rather than borrowing a neighbour's code: adaptive median
        driven as 2'b01 would tell the hardware to run a plain 3x3 median and
        the two paths would silently disagree about what was applied.
        """
        return CONTROL_CODE.get(self.filter_name)

    @property
    def hardware(self) -> str:
        """Where this decision can run.

        ``"rtl"`` — one pass of a filter the FPGA core implements, bit-exact.
        ``"rtl_multipass"`` — an RTL filter applied more than once. The top
        module streams one pass per frame; a second pass needs a cascaded core
        or a second stream, and the Wiener one-grey-level tolerance applies per
        pass.
        ``"software"`` — no RTL counterpart at all.
        """
        if self.filter_name in SOFTWARE_FILTERS:
            return "software"
        return "rtl" if self.passes == 1 else "rtl_multipass"


def select_filter(noise_class: str) -> str:
    """Return the filter for *noise_class*.

    Args:
        noise_class: One of :data:`denoising.config.CLASSES`.

    Returns:
        One of :data:`FILTERS`.

    Raises:
        ValueError: if *noise_class* is not a known class. There is no default:
            a class nobody mapped is a bug to fix, not a bypass to fall into.
    """
    try:
        return FILTER_FOR_CLASS[noise_class]
    except KeyError:
        raise ValueError(
            f"unknown noise class {noise_class!r}; expected one of {tuple(CLASSES)}"
        ) from None


def control_code(filter_name: str) -> int:
    """Return the 2-bit RTL control code for *filter_name*.

    Raises:
        ValueError: if *filter_name* is not one of :data:`FILTERS`.
    """
    try:
        return CONTROL_CODE[filter_name]
    except KeyError:
        raise ValueError(
            f"unknown filter {filter_name!r}; expected one of {FILTERS}"
        ) from None


def decide_filter(
    noise_class: str,
    confidence: float | None = None,
    *,
    threshold: float | None = None,
    fallback: str = "bypass",
    severity: str | None = None,
) -> FilterDecision:
    """Choose a filter, applying the low-confidence fallback if needed.

    Args:
        noise_class: The predicted class.
        confidence: Classifier confidence in [0, 1], or ``None`` when the class
            was chosen by a person — a manual choice has no confidence, and
            calling that 1.0 would be inventing a measurement.
        threshold: Confidence below which *fallback* is used. ``None`` disables
            the check.
        fallback: Filter to use below the threshold.
        severity: One of :data:`SEVERITY_LEVELS`, or ``None`` to use the plain
            class mapping exactly as before severity existed. Ignored for the
            clean class, which has no severity.

    Returns:
        A :class:`FilterDecision` recording both what was chosen and why.

    Raises:
        ValueError: if the class is unknown, the fallback is not a filter, the
            confidence is outside [0, 1], or the severity is not a level.
    """
    if severity is not None and severity not in SEVERITY_LEVELS:
        raise ValueError(f"unknown severity {severity!r}; expected one of {SEVERITY_LEVELS}")
    mapped = select_filter(noise_class)
    step = FilterStep(mapped, 1)
    used_severity: str | None = None
    if severity is not None and noise_class in SEVERITY_POLICY:
        step = SEVERITY_POLICY[noise_class][severity]
        used_severity = severity
    if fallback not in FILTERS:
        raise ValueError(f"unknown fallback filter {fallback!r}; expected one of {FILTERS}")
    if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")

    low = (
        confidence is not None
        and threshold is not None
        and float(confidence) < float(threshold)
    )
    # The fallback is one pass of an RTL filter and ignores severity: a
    # prediction we do not trust cannot select a strength either.
    return FilterDecision(
        filter_name=fallback if low else step.filter_name,
        noise_class=noise_class,
        confidence=None if confidence is None else float(confidence),
        used_fallback=bool(low),
        mapped_filter=step.filter_name,
        threshold=None if threshold is None else float(threshold),
        passes=1 if low else step.passes,
        severity=None if low else used_severity,
    )
