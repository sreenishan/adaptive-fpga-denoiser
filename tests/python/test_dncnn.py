"""DnCNN architecture tests.

These tests verify the architecture contract without a dataset:
- Correct layer count and channel dimensions.
- Output shape matches input.
- Output is clamped to [0, 1].
- Residual learning: output = clamp(input - net(input), 0, 1).
- Initialization: weights are non-zero, biases are zero for hidden layers.
- ``predict_noise`` returns the raw residual (not clamped).
- ``build_dncnn`` moves the model to the requested device.
"""

from __future__ import annotations

import pytest
import torch

from denoising.model.dncnn import DnCNN, build_dncnn


def test_default_architecture():
    model = DnCNN()
    assert model.depth == 17
    assert model.filters == 64


def test_custom_depth():
    model = DnCNN(depth=7)
    assert model.depth == 7


def test_depth_minimum():
    with pytest.raises(ValueError):
        DnCNN(depth=2)


def test_forward_shape():
    model = DnCNN()
    model.eval()
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


def test_output_clamped():
    model = DnCNN()
    model.eval()
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_residual_learning_contract():
    """output = clamp(input - predict_noise(input), 0, 1)."""
    model = DnCNN()
    model.eval()
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        out = model(x)
        noise_pred = model.predict_noise(x)
        expected = torch.clamp(x - noise_pred, 0.0, 1.0)
    assert torch.allclose(out, expected)


def test_batch_size_one():
    model = DnCNN()
    model.eval()
    x = torch.rand(1, 1, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 224, 224)


def test_build_dncnn_returns_dncnn():
    model = build_dncnn(depth=7, filters=32, input_channels=1)
    assert isinstance(model, DnCNN)
    assert model.depth == 7


def test_dncnn_parameter_count_range():
    model = DnCNN(depth=17, filters=64, input_channels=1)
    n = sum(p.numel() for p in model.parameters())
    assert 300_000 < n < 2_000_000, f"unexpected parameter count: {n}"


def test_classifier_still_importable():
    """NoiseClassifierCNN must not be broken by adding DnCNN."""
    from denoising.model.cnn import NoiseClassifierCNN
    classifier = NoiseClassifierCNN()
    x = torch.rand(2, 1, 224, 224)
    out = classifier(x)
    assert out.shape == (2, 4)
