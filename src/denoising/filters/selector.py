"""Automatic filter selection (spec sections 15 and 34).

The mapping from noise class to filter lives **here and nowhere else**. It is
read by the Python pipeline and it defines the 2-bit code sent to
``rtl/control/filter_controller.sv``; a second copy somewhere else is how the
software and the hardware come to disagree about what "speckle" means while
both look correct in isolation.

```text
clean       -> bypass    (2'b00)
salt_pepper -> median    (2'b01)
gaussian    -> gaussian  (2'b10)
speckle     -> wiener    (2'b11)
```

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
    "FILTERS",
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


@dataclass(frozen=True)
class FilterDecision:
    """What was chosen, and whether the classifier's answer was used.

    Attributes:
        filter_name: The filter to apply, one of :data:`FILTERS`.
        noise_class: The predicted class.
        confidence: The classifier's confidence, or ``None`` when the class was
            supplied by hand rather than predicted.
        used_fallback: True when confidence was below the threshold and
            *filter_name* is the fallback rather than the mapped filter.
        mapped_filter: What the mapping would have chosen. Kept even when the
            fallback wins, so the report can say what was overridden.
        threshold: The threshold the confidence was compared against.
    """

    filter_name: str
    noise_class: str
    confidence: float | None
    used_fallback: bool
    mapped_filter: str
    threshold: float | None

    @property
    def control_code(self) -> int:
        """The 2-bit code for the RTL controller."""
        return CONTROL_CODE[self.filter_name]


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

    Returns:
        A :class:`FilterDecision` recording both what was chosen and why.

    Raises:
        ValueError: if the class is unknown, the fallback is not a filter, or
            the confidence is outside [0, 1].
    """
    mapped = select_filter(noise_class)
    if fallback not in FILTERS:
        raise ValueError(f"unknown fallback filter {fallback!r}; expected one of {FILTERS}")
    if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")

    low = (
        confidence is not None
        and threshold is not None
        and float(confidence) < float(threshold)
    )
    return FilterDecision(
        filter_name=fallback if low else mapped,
        noise_class=noise_class,
        confidence=None if confidence is None else float(confidence),
        used_fallback=bool(low),
        mapped_filter=mapped,
        threshold=None if threshold is None else float(threshold),
    )
