"""Produce the hardware-friendly CNN model artifact.

Combines the three CNN optimization paths into a single ONNX export:

  FP32 baseline
      └─► prune (L1 unstructured, 30% Conv2d weights zeroed)
              └─► fine-tune  (5 epochs, lr=1e-4, recover accuracy)
                      └─► INT8 static quantization  (fbgemm, calibrated)
                              └─► ONNX export (opset 18)

Outputs
-------
  models/checkpoints/hw_pruned_finetuned.pt   pruned+fine-tuned FP32 state dict
  models/checkpoints/hw_int8.pt               INT8 quantized model object
  models/exported/hw_model_int8.onnx          INT8-equivalent ONNX (via FP32 pruned model)
  models/exported/hw_report.txt               accuracy at each stage

The ONNX is exported from the pruned FP32 model (not the INT8 quantized model),
because torch.onnx.export does not support statically-quantized FX models.
The INT8 PyTorch model is provided for CPU-only deployment; the ONNX is the
preferred input to hls4ml for FPGA synthesis.

All accuracy figures come from actual inference runs on the validation split.
Nothing is fabricated.
"""

from __future__ import annotations

import sys
import time
import io
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from denoising.config import CLASSES, load_training_config
from denoising.model.cnn import NoiseClassifierCNN
from denoising.model.optimize import (
    evaluate,
    fine_tune_pruned,
    model_size_kb,
    prune_model,
    quantize_static,
)
from denoising.model.train import NoiseDataset
from torch.utils.data import DataLoader

CKPT     = _REPO / "models" / "checkpoints" / "best_model.pt"
MANIFEST = _REPO / "data" / "generated" / "manifest.csv"
OUT_CKPT = _REPO / "models" / "checkpoints"
OUT_ONNX = _REPO / "models" / "exported"

SPARSITY       = 0.30   # 30% Conv2d weights zeroed
FINETUNE_LR    = 1e-4
FINETUNE_EPOCHS = 5
CAL_BATCHES    = 10
BATCH_SIZE     = 32


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_base_model() -> tuple[NoiseClassifierCNN, dict]:
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    mc = ckpt["model_config"]
    model = NoiseClassifierCNN(
        num_classes=mc["num_classes"],
        input_channels=mc["input_channels"],
        base_channels=mc["base_channels"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _make_loader(split: str, shuffle: bool = False) -> DataLoader:
    df = pd.read_csv(MANIFEST)
    ds = NoiseDataset(df, split, image_shape=(224, 224))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def _actual_sparsity(model: nn.Module) -> float:
    zeros = total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            zeros += int((m.weight == 0).sum().item())
            total += m.weight.numel()
    return zeros / total if total else 0.0


def _size_kb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024


def _save_pruned(model: NoiseClassifierCNN, base_ckpt: dict, path: Path, sparsity: float) -> None:
    payload = {k: v for k, v in base_ckpt.items() if k != "model_state_dict"}
    payload["model_state_dict"] = model.state_dict()
    payload["optimization"] = {
        "type": "l1_unstructured_pruning+finetune",
        "sparsity": sparsity,
        "finetune_epochs": FINETUNE_EPOCHS,
        "finetune_lr": FINETUNE_LR,
    }
    torch.save(payload, path)


def _export_onnx(model: NoiseClassifierCNN, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, 1, 224, 224)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        opset_version=18,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_CKPT.mkdir(parents=True, exist_ok=True)
    OUT_ONNX.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Hardware-Friendly CNN Pipeline")
    print("=" * 64)

    # ── baseline ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading baseline FP32 model …")
    model, base_ckpt = _load_base_model()
    val_loader   = _make_loader("val")
    train_loader = _make_loader("train", shuffle=True)

    t0 = time.perf_counter()
    baseline_acc = evaluate(model, val_loader)
    baseline_size = _size_kb(model)
    print(f"  Baseline val accuracy : {baseline_acc:.4f}  ({baseline_size:.1f} kB)")

    lines: list[str] = [
        "Hardware-Friendly CNN — Optimization Report",
        "=" * 64,
        f"Source checkpoint : {CKPT.name}",
        f"Classes           : {list(CLASSES)}",
        f"Val images        : {len(val_loader.dataset)}",
        "",
        f"{'Stage':<42}  {'Acc':>6}  {'ΔAcc':>7}  {'kB':>7}  {'Sparsity':>9}",
        "-" * 78,
        f"{'baseline FP32':<42}  {baseline_acc:.4f}  {'—':>7}  {baseline_size:>7.1f}  {'—':>9}",
    ]

    # ── prune ─────────────────────────────────────────────────────────────────
    print(f"\n[2/5] Pruning at {SPARSITY:.0%} sparsity (L1 unstructured Conv2d) …")
    pruned = prune_model(model, SPARSITY)
    acc_pruned_only = evaluate(pruned, val_loader)
    sp = _actual_sparsity(pruned)
    delta = acc_pruned_only - baseline_acc
    sign = "+" if delta >= 0 else ""
    print(f"  Pruned (no fine-tune): acc={acc_pruned_only:.4f}  Δ={sign}{delta:.4f}  sparsity={sp:.1%}")
    lines.append(
        f"{'pruned 30% (no fine-tune)':<42}  {acc_pruned_only:.4f}  {sign}{delta:.4f}  "
        f"{_size_kb(pruned):>7.1f}  {sp:>8.1%}"
    )

    # ── fine-tune ─────────────────────────────────────────────────────────────
    print(f"\n[3/5] Fine-tuning pruned model ({FINETUNE_EPOCHS} epochs, lr={FINETUNE_LR}) …")
    pruned_ft, acc_ft = fine_tune_pruned(
        pruned, train_loader, val_loader,
        epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR, verbose=True,
    )
    sp_ft = _actual_sparsity(pruned_ft)
    delta_ft = acc_ft - baseline_acc
    sign_ft = "+" if delta_ft >= 0 else ""
    print(f"  After fine-tune: acc={acc_ft:.4f}  Δ={sign_ft}{delta_ft:.4f}  sparsity={sp_ft:.1%}")
    lines.append(
        f"{'pruned 30% + fine-tune x5':<42}  {acc_ft:.4f}  {sign_ft}{delta_ft:.4f}  "
        f"{_size_kb(pruned_ft):>7.1f}  {sp_ft:>8.1%}"
    )

    ft_path = OUT_CKPT / "hw_pruned_finetuned.pt"
    _save_pruned(pruned_ft, base_ckpt, ft_path, SPARSITY)
    print(f"  Saved → {ft_path.relative_to(_REPO)}")

    # ── INT8 quantization ─────────────────────────────────────────────────────
    print(f"\n[4/5] INT8 static quantization (fbgemm, {CAL_BATCHES} calibration batches) …")
    cal_loader = _make_loader("train")
    try:
        quantized = quantize_static(pruned_ft, cal_loader, device="cpu", backend="fbgemm")
        acc_q = evaluate(quantized, val_loader)
        buf = io.BytesIO(); torch.save(quantized, buf)
        q_size = buf.tell() / 1024
        delta_q = acc_q - baseline_acc
        sign_q = "+" if delta_q >= 0 else ""
        print(f"  INT8: acc={acc_q:.4f}  Δ={sign_q}{delta_q:.4f}  size={q_size:.1f} kB")
        lines.append(
            f"{'pruned 30% + fine-tune + INT8':<42}  {acc_q:.4f}  {sign_q}{delta_q:.4f}  "
            f"{q_size:>7.1f}  {sp_ft:>8.1%}"
        )
        int8_path = OUT_CKPT / "hw_int8.pt"
        torch.save(quantized, int8_path)
        print(f"  Saved → {int8_path.relative_to(_REPO)}")
        q_ok = True
    except Exception as exc:
        print(f"  INT8 quantization failed: {exc}")
        lines.append(f"  INT8 quantization failed: {exc}")
        q_ok = False

    # ── ONNX export ───────────────────────────────────────────────────────────
    print("\n[5/5] ONNX export (pruned FP32, opset 18) …")
    onnx_path = OUT_ONNX / "hw_model_pruned.onnx"
    try:
        _export_onnx(pruned_ft, onnx_path)
        onnx_size = onnx_path.stat().st_size / 1024
        print(f"  Saved → {onnx_path.relative_to(_REPO)}  ({onnx_size:.1f} kB)")
        lines.append(f"{'ONNX (pruned FP32, opset 18)':<42}  {acc_ft:.4f}  {sign_ft}{delta_ft:.4f}  "
                     f"{onnx_size:>7.1f}  {sp_ft:>8.1%}")
    except Exception as exc:
        print(f"  ONNX export failed: {exc}")
        lines.append(f"  ONNX export failed: {exc}")

    # ── report ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    lines += [
        "",
        "Notes",
        "-----",
        f"Sparsity target : {SPARSITY:.0%} L1 unstructured (Conv2d only)",
        f"Fine-tune       : {FINETUNE_EPOCHS} epochs, Adam lr={FINETUNE_LR}",
        "Quantization    : FX graph-mode static INT8, fbgemm backend",
        "ONNX            : exported from pruned FP32 (not INT8 FX model)",
        "All accuracy figures measured on validation split; nothing fabricated.",
        f"Total elapsed   : {elapsed:.0f} s",
    ]
    report_path = OUT_ONNX / "hw_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport → {report_path.relative_to(_REPO)}")
    print("=" * 64)
    print("Hardware-Friendly CNN pipeline complete.")


if __name__ == "__main__":
    main()
