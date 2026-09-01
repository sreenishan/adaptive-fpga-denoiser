"""Software reference filters and the filter selector (spec sections 14-15).

These implementations are the **golden reference**: the RTL in ``rtl/filters/``
is verified against them pixel by pixel, not the other way round.

All three operate on a 3x3 neighbourhood with **replicated edges**, the policy
``rtl/common/window_3x3.sv`` implements. Two of them are exact integer
arithmetic and are required to match the hardware bit for bit; the Wiener
filter needs a division and is allowed one grey level.

```python
from denoising.filters import apply_filter, select_filter

out = apply_filter(image, select_filter("salt_pepper"))
```
"""

from __future__ import annotations

from .gaussian import (
    BINOMIAL_DIVISOR_3X3,
    BINOMIAL_EFFECTIVE_SIGMA,
    BINOMIAL_KERNEL_3X3,
    gaussian_filter,
    gaussian_kernel,
)
from .median import median_filter
from .selector import (
    CONTROL_CODE,
    FILTER_FOR_CLASS,
    FILTERS,
    FilterDecision,
    control_code,
    decide_filter,
    select_filter,
)
from .wiener import estimate_noise_variance, wiener_filter

__all__ = [
    "BINOMIAL_DIVISOR_3X3",
    "BINOMIAL_EFFECTIVE_SIGMA",
    "BINOMIAL_KERNEL_3X3",
    "CONTROL_CODE",
    "FILTERS",
    "FILTER_FOR_CLASS",
    "FilterDecision",
    "apply_filter",
    "control_code",
    "decide_filter",
    "estimate_noise_variance",
    "gaussian_filter",
    "gaussian_kernel",
    "median_filter",
    "select_filter",
    "wiener_filter",
]


def apply_filter(image, filter_name: str, **parameters):
    """Apply the filter named *filter_name* to *image*.

    One dispatch point, so a caller never grows its own if-chain that can drift
    from :data:`FILTERS`.

    Args:
        image: 2-D uint8 grayscale image.
        filter_name: One of :data:`FILTERS`. ``"bypass"`` returns a copy — the
            clean class is a real decision to do nothing, not a missing case.
        **parameters: Passed to the chosen filter (``kernel_size``, ``sigma``,
            ``noise_variance``). Parameters that do not apply are ignored, so
            one configuration block can drive all four branches.

    Returns:
        A new uint8 array.

    Raises:
        ValueError: if *filter_name* is not one of :data:`FILTERS`.
    """
    from ..noise._common import validate_image

    image = validate_image(image)
    kernel_size = parameters.get("kernel_size", 3)

    if filter_name == "bypass":
        return image.copy()
    if filter_name == "median":
        return median_filter(image, kernel_size)
    if filter_name == "gaussian":
        return gaussian_filter(
            image,
            kernel_size,
            parameters.get("sigma"),
            integer_kernel=parameters.get("integer_kernel", True),
        )
    if filter_name == "wiener":
        return wiener_filter(image, kernel_size, parameters.get("noise_variance"))
    raise ValueError(f"unknown filter {filter_name!r}; expected one of {FILTERS}")
