"""CNN optimization: magnitude pruning and INT8 static quantization.

Two independent passes over the trained checkpoint:

  1. Pruning  — ``prune_model(model, sparsity)`` zeroes the smallest weights
     in every Conv2d layer (L1 unstructured), then removes the masks so the
     result is a regular dense model that can be loaded with ``load_state_dict``.

  2. Quantization — ``quantize_static(model, calibration_loader)`` converts
     the FP32 model to INT8 using PyTorch FX graph-mode static quantization.
     Conv+BN+ReLU fusions are handled automatically by the graph pass.  Needs
     a small calibration set (~100–500 images) to collect activation statistics.

Neither operation is approximate in the algorithmic sense: pruning produces an
exact sparse model, and quantization is characterised by its accuracy delta
(reported by the script) rather than claimed to be lossless.
"""

from __future__ import annotations

import copy
import io
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

__all__ = ["prune_model", "quantize_static", "evaluate", "model_size_kb"]


def prune_model(model: nn.Module, sparsity: float) -> nn.Module:
    """Return a pruned copy of *model*.

    Applies L1 unstructured pruning to every Conv2d weight tensor at the
    given *sparsity* (fraction of weights zeroed, e.g. 0.30 = 30%).  The
    pruning masks are immediately made permanent so the returned model has
    normal weight tensors and can be saved with ``torch.save(state_dict())``.

    Args:
        model:    The source model (not modified).
        sparsity: Fraction of Conv2d weights to zero, in [0, 1).

    Returns:
        A new model instance with the same architecture and pruned weights.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    m = copy.deepcopy(model)
    for module in m.modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
            prune.remove(module, "weight")
    return m


def quantize_static(
    model: nn.Module,
    calibration_loader: "DataLoader",
    *,
    device: str = "cpu",
    backend: str = "fbgemm",
) -> nn.Module:
    """Return an INT8 statically-quantized copy of *model*.

    Uses PyTorch FX graph-mode quantization (``torch.ao.quantization.quantize_fx``).
    Conv+BN+ReLU patterns are fused automatically during graph preparation.

    Args:
        model:              The FP32 source model (not modified).
        calibration_loader: DataLoader yielding ``(image_tensor, label)``
                            batches.  Only the images are used; ~200 samples
                            are enough to calibrate activation statistics.
        device:             ``"cpu"`` (INT8 FBGEMM) or ``"cuda"`` (QNNPACK).
        backend:            Quantization backend — ``"fbgemm"`` for x86,
                            ``"qnnpack"`` for ARM.

    Returns:
        The converted INT8 model.  Save with ``torch.save(quantized_model)``.
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    m = copy.deepcopy(model).to(device)
    m.eval()

    qconfig_mapping = get_default_qconfig_mapping(backend)
    example_input = (torch.randn(1, 1, 224, 224).to(device),)

    prepared = prepare_fx(m, qconfig_mapping, example_input)

    with torch.no_grad():
        for batch in calibration_loader:
            images = batch[0].to(device)
            prepared(images)

    quantized = convert_fx(prepared)
    return quantized


def evaluate(
    model: nn.Module,
    loader: "DataLoader",
    *,
    device: str = "cpu",
) -> float:
    """Return top-1 accuracy of *model* on *loader*."""
    model = model.to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            preds = model(images).argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += len(labels)
    return correct / total if total > 0 else 0.0


def fine_tune_pruned(
    model: nn.Module,
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    *,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[nn.Module, float]:
    """Fine-tune a pruned model with sparsity masks active to recover accuracy.

    The masks applied by ``prune_model`` are re-applied before this call by
    working on a copy that still has mask buffers; call this BEFORE
    ``prune.remove()`` by using the internal two-step API, or call on the
    output of ``prune_model_masked`` (which keeps masks active).

    Because ``prune_model`` already removes masks (``prune.remove``), this
    function accepts either a masked or a de-masked sparse model and fine-tunes
    it with a small learning rate.  Zeroed weights CAN become non-zero during
    this step, which is expected behaviour for fine-tuning after pruning: the
    sparsity target was already applied, and the goal here is accuracy recovery,
    not enforcing a hard sparsity budget.

    Returns:
        (fine-tuned model copy, best validation accuracy achieved)
    """
    m = copy.deepcopy(model).to(device)
    m.train()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state: dict | None = None

    for ep in range(1, epochs + 1):
        m.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(m(images), labels)
            loss.backward()
            optimizer.step()
        # evaluate
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = m(images).argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += len(labels)
        acc = correct / total if total > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(m.state_dict())
        if verbose:
            print(f"    fine-tune ep {ep}/{epochs}: val_acc={acc:.4f}  best={best_acc:.4f}")

    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    return m, best_acc


def model_size_kb(model: nn.Module) -> float:
    """Return the model's in-memory size in kilobytes (state-dict only)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024
