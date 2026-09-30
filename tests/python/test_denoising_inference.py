"""DnCNN inference and pipeline tests.

These tests verify the inference wrapper and denoising pipeline without
requiring a trained checkpoint.  A minimal DnCNN is created and saved to a
temp file, then loaded via ``DnCNNInferencer.from_checkpoint``.

Tests:
- Checkpoint round-trip: save → load → denoise produces uint8 output.
- Output shape matches input.
- Output dtype is uint8.
- Output values are in [0, 255].
- ``load_denoiser`` with an explicit path works.
- ``load_denoiser`` with a missing path raises FileNotFoundError.
- ``process_image_denoiser`` returns a PipelineResult with the right fields.
- The existing ``process_image`` is not broken.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from denoising.model.dncnn import build_dncnn
from denoising.model.inference_denoiser import DnCNNInferencer, load_denoiser


# ─── Shared fixture: tiny trained checkpoint ──────────────────────────────────


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ckpt")
    path = tmp / "denoiser.pt"
    model = build_dncnn(depth=3, filters=8, input_channels=1)
    torch.save(
        {
            "model_type": "DnCNN",
            "depth": model.depth,
            "filters": model.filters,
            "epoch": 0,
            "train_loss": 0.1,
            "val_loss": 0.1,
            "val_psnr": 20.0,
            "config": {},
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def inferencer(checkpoint_path):
    return DnCNNInferencer.from_checkpoint(checkpoint_path, device="cpu")


@pytest.fixture()
def gray_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_denoise_returns_uint8(inferencer, gray_image):
    result = inferencer.denoise(gray_image)
    assert result.dtype == np.uint8


def test_denoise_shape_preserved(inferencer, gray_image):
    result = inferencer.denoise(gray_image)
    assert result.shape == gray_image.shape


def test_denoise_values_in_range(inferencer, gray_image):
    result = inferencer.denoise(gray_image)
    assert result.min() >= 0
    assert result.max() <= 255


def test_load_denoiser_from_path(checkpoint_path, gray_image):
    inf = load_denoiser(checkpoint_path, device="cpu")
    result = inf.denoise(gray_image)
    assert result.shape == gray_image.shape


def test_load_denoiser_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_denoiser(Path("/nonexistent/denoiser.pt"))


def test_metadata_keys(inferencer):
    meta = inferencer.metadata
    assert "model_type" in meta
    assert meta["model_type"] == "DnCNN"
    assert "depth" in meta
    assert "filters" in meta


def test_process_image_denoiser_returns_pipeline_result(checkpoint_path, gray_image):
    from denoising.pipeline.denoising_pipeline import process_image_denoiser

    result = process_image_denoiser(gray_image, checkpoint=checkpoint_path)
    assert result.selected_filter == "dncnn"
    assert result.noise_class == "dncnn"
    assert result.output.shape == gray_image.shape
    assert result.output.dtype == np.uint8
    assert result.metrics is None  # no clean reference supplied
    assert result.metrics_note is not None


def test_process_image_denoiser_with_reference(checkpoint_path, gray_image):
    from denoising.pipeline.denoising_pipeline import process_image_denoiser

    clean = np.full_like(gray_image, 128)
    result = process_image_denoiser(gray_image, checkpoint=checkpoint_path, clean_reference=clean)
    assert result.metrics is not None
    assert hasattr(result.metrics, "psnr")
    assert hasattr(result.metrics, "ssim")
    assert hasattr(result.metrics, "mse")


def test_existing_process_image_not_broken():
    """The classifier-based pipeline must still be importable and callable."""
    from denoising.config import load_inference_config
    from denoising.pipeline.adaptive_pipeline import process_image

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    config = load_inference_config()
    result = process_image(img, config)
    assert result.output.shape == img.shape
    assert result.selected_filter != ""
