"""Tests for src/denoising/model/optimize.py"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from denoising.model.cnn import NoiseClassifierCNN
from denoising.model.optimize import (
    evaluate,
    model_size_kb,
    prune_model,
    quantize_static,
)


# ─── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tiny_model() -> NoiseClassifierCNN:
    return NoiseClassifierCNN(num_classes=5, input_channels=1, base_channels=4, dropout=0.0)


def _rand_loader(n: int = 64, num_classes: int = 5) -> DataLoader:
    images = torch.randn(n, 1, 32, 32)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=16)


# ─── prune_model ──────────────────────────────────────────────────────────────


def test_prune_returns_new_instance(tiny_model):
    pruned = prune_model(tiny_model, sparsity=0.30)
    assert pruned is not tiny_model


def test_prune_does_not_modify_original(tiny_model):
    w_before = tiny_model.features[0].block[0].weight.clone()
    prune_model(tiny_model, sparsity=0.50)
    assert torch.equal(tiny_model.features[0].block[0].weight, w_before)


def test_prune_achieves_target_sparsity(tiny_model):
    sparsity = 0.40
    pruned = prune_model(tiny_model, sparsity)
    zeros = total = 0
    for m in pruned.modules():
        if isinstance(m, nn.Conv2d):
            zeros += int((m.weight == 0).sum().item())
            total += m.weight.numel()
    actual = zeros / total
    # L1 prune rounds to the nearest element; allow ±5% tolerance
    assert abs(actual - sparsity) < 0.05, f"sparsity={actual:.3f}, expected ~{sparsity}"


def test_pruned_model_has_no_mask_buffers(tiny_model):
    pruned = prune_model(tiny_model, sparsity=0.30)
    for name, _ in pruned.named_buffers():
        assert "mask" not in name, f"Pruning mask not removed: {name}"


def test_prune_state_dict_loadable(tiny_model):
    pruned = prune_model(tiny_model, sparsity=0.30)
    fresh = NoiseClassifierCNN(num_classes=5, input_channels=1, base_channels=4, dropout=0.0)
    fresh.load_state_dict(pruned.state_dict())  # must not raise


def test_prune_invalid_sparsity(tiny_model):
    with pytest.raises(ValueError):
        prune_model(tiny_model, sparsity=1.0)
    with pytest.raises(ValueError):
        prune_model(tiny_model, sparsity=-0.1)


def test_prune_zero_sparsity_unchanged(tiny_model):
    pruned = prune_model(tiny_model, sparsity=0.0)
    for name, p in tiny_model.named_parameters():
        if "weight" in name:
            p2 = dict(pruned.named_parameters())[name]
            assert torch.equal(p, p2), f"Weights changed at zero sparsity: {name}"


# ─── evaluate ─────────────────────────────────────────────────────────────────


def test_evaluate_returns_float_in_range(tiny_model):
    loader = _rand_loader()
    acc = evaluate(tiny_model, loader)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_evaluate_perfect_model():
    """A model that always predicts class 0 on a dataset where all labels=0."""
    class AlwaysZero(nn.Module):
        def forward(self, x):
            b = x.shape[0]
            out = torch.zeros(b, 5)
            out[:, 0] = 10.0
            return out

    images = torch.randn(20, 1, 32, 32)
    labels = torch.zeros(20, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=10)
    acc = evaluate(AlwaysZero(), loader)
    assert acc == pytest.approx(1.0)


# ─── model_size_kb ────────────────────────────────────────────────────────────


def test_model_size_kb_positive(tiny_model):
    size = model_size_kb(tiny_model)
    assert size > 0


def test_larger_model_has_larger_size():
    small = NoiseClassifierCNN(num_classes=5, input_channels=1, base_channels=4, dropout=0.0)
    large = NoiseClassifierCNN(num_classes=5, input_channels=1, base_channels=16, dropout=0.0)
    assert model_size_kb(large) > model_size_kb(small)


# ─── quantize_static ──────────────────────────────────────────────────────────


def test_quantize_static_runs(tiny_model):
    loader = _rand_loader(n=32)
    q = quantize_static(tiny_model, loader, device="cpu", backend="fbgemm")
    assert q is not None


def test_quantize_does_not_modify_original(tiny_model):
    w_before = tiny_model.features[0].block[0].weight.clone()
    loader = _rand_loader(n=16)
    quantize_static(tiny_model, loader, device="cpu", backend="fbgemm")
    assert torch.equal(tiny_model.features[0].block[0].weight, w_before)


def test_quantize_output_shape(tiny_model):
    loader = _rand_loader(n=16)
    q = quantize_static(tiny_model, loader, device="cpu")
    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        out = q(x)
    assert out.shape == (2, 5)


def test_quantized_model_saveable(tiny_model):
    loader = _rand_loader(n=16)
    q = quantize_static(tiny_model, loader, device="cpu")
    buf = io.BytesIO()
    torch.save(q, buf)
    assert buf.tell() > 0
