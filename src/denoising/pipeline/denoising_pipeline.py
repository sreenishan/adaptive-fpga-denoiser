"""DnCNN denoising pipeline — mirrors :func:`~denoising.pipeline.adaptive_pipeline.process_image`.

``process_image_denoiser`` accepts the same arguments as ``process_image`` and
returns a compatible :class:`~denoising.pipeline.adaptive_pipeline.PipelineResult`
so the evaluation layer can treat them interchangeably.

The existing ``process_image`` function is **not modified**.  Both functions
share ``PipelineResult`` and ``ImageQuality`` but nothing else.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..metrics.image_quality import calculate_quality
from ..noise._common import GrayImage
from ..pipeline.adaptive_pipeline import PipelineResult

__all__ = ["process_image_denoiser"]


def process_image_denoiser(
    image: GrayImage,
    checkpoint: Path | str | None = None,
    clean_reference: GrayImage | None = None,
    *,
    device: str | None = None,
) -> PipelineResult:
    """Denoise *image* using the DnCNN checkpoint and return a PipelineResult.

    Args:
        image:           Noisy uint8 grayscale image.
        checkpoint:      Path to ``denoiser.pt``; uses the default project path
                         when ``None``.
        clean_reference: Optional clean original for PSNR/SSIM metrics.
        device:          PyTorch device string; ``None`` → auto.

    Returns:
        :class:`~denoising.pipeline.adaptive_pipeline.PipelineResult` with
        ``noise_class="dncnn"``, ``selected_filter="dncnn"``, and metrics
        populated when *clean_reference* is supplied.

    Raises:
        FileNotFoundError: When *checkpoint* is ``None`` and the default
            checkpoint path does not exist.
    """
    from ..filters.selector import FilterDecision
    from ..model.inference_denoiser import load_denoiser

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    inferencer = load_denoiser(checkpoint, device=device)
    t_load_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    denoised = inferencer.denoise(image)
    t_denoise_ms = (time.perf_counter() - t0) * 1000.0

    metrics = None
    noisy_metrics = None
    metrics_note = None
    if clean_reference is not None:
        metrics = calculate_quality(denoised, clean_reference)
        noisy_metrics = calculate_quality(image, clean_reference)
    else:
        metrics_note = "no clean reference supplied"

    decision = FilterDecision(
        filter_name="dncnn",
        noise_class="dncnn",
        confidence=None,
        used_fallback=False,
        mapped_filter="dncnn",
        threshold=None,
        passes=1,
        severity=None,
    )

    return PipelineResult(
        noise_class="dncnn",
        confidence=None,
        selected_filter="dncnn",
        decision=decision,
        input=image,
        output=denoised,
        metrics=metrics,
        metrics_note=metrics_note,
        noisy_metrics=noisy_metrics,
        timings_ms={
            "load_checkpoint_ms": round(t_load_ms, 1),
            "denoise_ms": round(t_denoise_ms, 1),
            "total_ms": round((time.perf_counter() - t_total) * 1000.0, 1),
        },
        severity=None,
    )
