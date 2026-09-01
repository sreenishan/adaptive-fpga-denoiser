"""The end-to-end adaptive pipeline (spec section 16).

```text
input -> preprocess -> classify -> select filter -> filter -> result
```

One call, :func:`process_image`, joins the pieces the rest of the package
provides. The classifier is injected rather than imported, so the pipeline works
today with the class chosen by hand and works unchanged once phase 6 provides a
trained model — the UI on top never learns which of the two happened.

Two honesty rules are enforced here rather than left to callers:

**A manual choice has no confidence.** When a person picks the class, the
result's ``confidence`` is ``None``. Recording 1.0 would put a
measurement-shaped number where no measurement happened, and the display would
show a certainty nobody claimed.

**Quality metrics need a clean reference.** MSE, PSNR and SSIM all compare
against an original. Given a noisy photograph and no original, the result
carries ``metrics = None`` and ``metrics_note`` saying why. Comparing the output
to its own noisy input would produce three numbers that describe how much the
filter changed and nothing whatsoever about quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ..config import CLASSES, InferenceConfig
from ..filters import apply_filter, decide_filter
from ..filters.selector import FilterDecision
from ..metrics.image_quality import ImageQuality, calculate_quality
from ..noise._common import GrayImage, validate_image
from ..preprocessing import preprocess

__all__ = ["NoiseClassifier", "PipelineResult", "process_image"]


class NoiseClassifier(Protocol):
    """What the pipeline needs from a classifier.

    Phase 6 supplies an implementation backed by a trained CNN. Anything with
    this method works — including a stub in a test.
    """

    def predict(self, image: GrayImage) -> tuple[str, float]:
        """Return ``(noise_class, confidence)`` for *image*."""
        ...


@dataclass(frozen=True)
class PipelineResult:
    """Everything one run produced.

    Attributes:
        noise_class: The class the filter choice was based on.
        confidence: Classifier confidence, or ``None`` for a manual choice.
        selected_filter: The filter actually applied.
        decision: The full selection record, including whether the
            low-confidence fallback was used.
        input: The image as it entered the pipeline, uint8.
        output: The denoised image, uint8.
        metrics: Quality against a clean reference, or ``None`` when no
            reference was supplied.
        metrics_note: Why metrics are absent, when they are.
        noisy_metrics: Quality of the *input* against the reference, so the
            improvement the filter made can be stated rather than implied.
        timings_ms: Wall-clock milliseconds for each stage.
    """

    noise_class: str
    confidence: float | None
    selected_filter: str
    decision: FilterDecision
    input: GrayImage
    output: GrayImage
    metrics: ImageQuality | None = None
    metrics_note: str | None = None
    noisy_metrics: ImageQuality | None = None
    timings_ms: Mapping[str, float] = field(default_factory=dict)

    @property
    def psnr_improvement(self) -> float | None:
        """Denoised PSNR minus noisy PSNR, or ``None`` without a reference.

        ``None`` also when either PSNR is infinite: subtracting infinities is
        not an improvement figure.
        """
        if self.metrics is None or self.noisy_metrics is None:
            return None
        import math

        if math.isinf(self.metrics.psnr) or math.isinf(self.noisy_metrics.psnr):
            return None
        return self.metrics.psnr - self.noisy_metrics.psnr

    def as_dict(self) -> dict[str, Any]:
        """The structured result of spec section 16, without the pixel arrays."""
        return {
            "noise_class": self.noise_class,
            "confidence": self.confidence,
            "selected_filter": self.selected_filter,
            "used_fallback": self.decision.used_fallback,
            "control_code": self.decision.control_code,
            "metrics": None
            if self.metrics is None
            else {
                "mse": self.metrics.mse,
                "psnr": self.metrics.psnr,
                "ssim": self.metrics.ssim,
            },
            "metrics_note": self.metrics_note,
            "psnr_improvement": self.psnr_improvement,
            "timings_ms": dict(self.timings_ms),
        }


def _filter_parameters(config: InferenceConfig, filter_name: str) -> dict[str, Any]:
    """The configured parameters for one filter."""
    filters = config.filters
    if filter_name == "median":
        return {"kernel_size": filters.median.kernel_size}
    if filter_name == "gaussian":
        return {
            "kernel_size": filters.gaussian.kernel_size,
            "sigma": filters.gaussian.sigma,
            "integer_kernel": True,
        }
    if filter_name == "wiener":
        return {
            "kernel_size": filters.wiener.kernel_size,
            "noise_variance": filters.wiener.noise_variance,
        }
    return {}


def process_image(
    image: GrayImage,
    config: InferenceConfig,
    *,
    noise_class: str | None = None,
    classifier: NoiseClassifier | None = None,
    reference: GrayImage | None = None,
    resize: bool = False,
) -> PipelineResult:
    """Classify *image*, choose a filter, apply it and measure the result.

    Args:
        image: 2-D uint8 grayscale image to denoise.
        config: Inference configuration — thresholds, filter parameters.
        noise_class: Class chosen by hand. Mutually exclusive with
            *classifier*; exactly one of the two must be given.
        classifier: Something implementing :class:`NoiseClassifier`.
        reference: The clean original, when there is one. Without it the result
            carries no quality metrics, because there is nothing to compare to.
        resize: Resize the image to the configured geometry before filtering.
            Off by default: the caller submitted an image and gets that image
            back, at its own resolution.

    Returns:
        A :class:`PipelineResult`.

    Raises:
        ValueError: if neither or both of *noise_class* and *classifier* are
            given, if the class is unknown, or if *reference* does not match
            the image being measured.
    """
    image = validate_image(image)
    if (noise_class is None) == (classifier is None):
        raise ValueError(
            "give exactly one of noise_class (a manual choice) or classifier "
            "(a prediction)"
        )

    timings: dict[str, float] = {}

    started = time.perf_counter()
    working = preprocess(image, config.image) if resize else image
    timings["preprocess"] = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    if classifier is not None:
        predicted, confidence = classifier.predict(working)
        if predicted not in CLASSES:
            raise ValueError(
                f"classifier returned unknown class {predicted!r}; "
                f"expected one of {tuple(CLASSES)}"
            )
    else:
        predicted, confidence = noise_class, None
        if predicted not in CLASSES:
            raise ValueError(
                f"unknown noise class {predicted!r}; expected one of {tuple(CLASSES)}"
            )
    timings["classify"] = (time.perf_counter() - started) * 1000.0

    decision = decide_filter(
        predicted,
        confidence,
        threshold=config.confidence.threshold,
        fallback=config.confidence.fallback,
    )

    started = time.perf_counter()
    output = apply_filter(
        working, decision.filter_name, **_filter_parameters(config, decision.filter_name)
    )
    timings["filter"] = (time.perf_counter() - started) * 1000.0
    timings["total"] = sum(timings.values())

    metrics: ImageQuality | None = None
    noisy_metrics: ImageQuality | None = None
    note: str | None = None
    if reference is None:
        note = (
            "no clean reference was supplied, so MSE, PSNR and SSIM cannot be "
            "computed; comparing the output to its own noisy input would "
            "measure change, not quality"
        )
    else:
        reference = validate_image(reference, "reference")
        if reference.shape != working.shape:
            raise ValueError(
                f"reference shape {reference.shape} does not match the image "
                f"being measured {working.shape}"
            )
        max_pixel = config.evaluation.max_pixel_value
        metrics = calculate_quality(reference, output, max_pixel)
        noisy_metrics = calculate_quality(reference, working, max_pixel)

    return PipelineResult(
        noise_class=predicted,
        confidence=confidence,
        selected_filter=decision.filter_name,
        decision=decision,
        input=working,
        output=output,
        metrics=metrics,
        metrics_note=note,
        noisy_metrics=noisy_metrics,
        timings_ms=timings,
    )
