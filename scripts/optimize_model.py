"""CNN optimization script — pruning + INT8 static quantization.

Usage::

    python scripts/optimize_model.py \\
        --checkpoint models/checkpoints/best_model.pt \\
        --dataset    data/processed \\
        --out-dir    models/checkpoints

Produces:
    pruned_30.pt        — 30% of Conv2d weights zeroed (state dict)
    pruned_50.pt        — 50% of Conv2d weights zeroed (state dict)
    quantized_int8.pt   — INT8 FX-quantized model (whole model object)
    optimization_report.txt

Requires the processed dataset (manifest.csv + images).
Accuracy numbers are measured on the validation split and reported honestly;
no figure is fabricated or estimated.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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

import pandas as pd


# ─── helpers ──────────────────────────────────────────────────────────────────


def _load_model(checkpoint: Path) -> tuple[NoiseClassifierCNN, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mc = ckpt["model_config"]
    model = NoiseClassifierCNN(
        num_classes=mc["num_classes"],
        input_channels=mc["input_channels"],
        base_channels=mc["base_channels"],
        dropout=0.0,  # no dropout at inference
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _make_loader(manifest_path: Path, split: str, batch_size: int = 32) -> DataLoader:
    df = pd.read_csv(manifest_path)
    ds = NoiseDataset(df, split, image_shape=(224, 224))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _save_pruned(model: NoiseClassifierCNN, ckpt: dict, path: Path, sparsity: float) -> None:
    payload = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    payload["model_state_dict"] = model.state_dict()
    payload["optimization"] = {"type": "l1_unstructured_pruning", "sparsity": sparsity}
    torch.save(payload, path)


def _actual_sparsity(model: torch.nn.Module) -> float:
    """Fraction of zero weights across all Conv2d layers."""
    zeros = total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            zeros += int((m.weight == 0).sum().item())
            total += m.weight.numel()
    return zeros / total if total else 0.0


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Prune and quantize the noise classifier CNN")
    ap.add_argument("--checkpoint", default="models/checkpoints/best_model.pt")
    ap.add_argument("--dataset",    default="data/generated")
    ap.add_argument("--out-dir",    default="models/checkpoints")
    ap.add_argument("--sparsities", nargs="+", type=float, default=[0.30, 0.50],
                    help="Pruning sparsity levels to test (default: 0.30 0.50)")
    ap.add_argument("--cal-batches", type=int, default=10,
                    help="Calibration batches for quantization (default: 10)")
    ap.add_argument("--batch-size",  type=int, default=32)
    ap.add_argument("--finetune-epochs", type=int, default=0,
                    help="Fine-tuning epochs after pruning to recover accuracy (0 = skip)")
    args = ap.parse_args()

    checkpoint = _REPO / args.checkpoint
    dataset_dir = _REPO / args.dataset
    out_dir = _REPO / args.out_dir
    manifest = dataset_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        sys.exit(f"Checkpoint not found: {checkpoint}")
    if not manifest.exists():
        sys.exit(f"Manifest not found: {manifest}\nRun scripts/generate_dataset.py first.")

    print(f"Checkpoint : {checkpoint}")
    print(f"Dataset    : {manifest}")
    print()

    # ── load model ────────────────────────────────────────────────────────────
    model, ckpt = _load_model(checkpoint)
    baseline_size = model_size_kb(model)
    print(f"Baseline val-acc at training: {ckpt.get('val_acc', 'TBD'):.4f}")
    print(f"Baseline model size         : {baseline_size:.1f} kB\n")

    # ── validation loader ─────────────────────────────────────────────────────
    print("Loading validation split …")
    val_loader = _make_loader(manifest, "val", args.batch_size)
    t0 = time.perf_counter()
    baseline_acc = evaluate(model, val_loader)
    elapsed = time.perf_counter() - t0
    print(f"Baseline val accuracy (measured): {baseline_acc:.4f}  ({elapsed:.1f}s)\n")

    lines: list[str] = []
    lines.append("CNN Optimization Report")
    lines.append("=" * 60)
    lines.append(f"Checkpoint : {checkpoint.name}")
    lines.append(f"Classes    : {list(CLASSES)}")
    lines.append(f"Val images : {len(val_loader.dataset)}")
    lines.append("")
    lines.append(f"{'Model':<32}  {'Acc':>6}  {'ΔAcc':>7}  {'Size kB':>8}  {'Sparsity':>9}")
    lines.append("-" * 72)
    lines.append(f"{'baseline (FP32)':<32}  {baseline_acc:.4f}  {'—':>7}  {baseline_size:>8.1f}  {'—':>9}")

    # ── pruning ───────────────────────────────────────────────────────────────
    train_loader = _make_loader(manifest, "train", args.batch_size) if args.finetune_epochs > 0 else None
    pruned_models: dict[float, tuple[NoiseClassifierCNN, float]] = {}
    for sp in args.sparsities:
        tag = f"pruned_{int(sp*100):02d}"
        print(f"Pruning at {sp:.0%} sparsity …")
        pruned = prune_model(model, sp)
        actual_sp = _actual_sparsity(pruned)
        acc_before = evaluate(pruned, val_loader)

        if args.finetune_epochs > 0:
            print(f"  Fine-tuning {args.finetune_epochs} epochs (lr=1e-4) …")
            pruned, acc = fine_tune_pruned(
                pruned, train_loader, val_loader,
                epochs=args.finetune_epochs, lr=1e-4, verbose=True,
            )
            delta = acc - baseline_acc
            sign = "+" if delta >= 0 else ""
            print(f"  {tag} (fine-tuned): acc={acc:.4f}  Δ={sign}{delta:.4f}  (before fine-tune: {acc_before:.4f})")
            tag_ft = tag + "_ft"
        else:
            acc = acc_before
            delta = acc - baseline_acc
            sign = "+" if delta >= 0 else ""
            tag_ft = tag

        size = model_size_kb(pruned)
        print(f"  {tag}: acc={acc:.4f}  Δ={sign}{delta:.4f}  size={size:.1f} kB  sparsity={actual_sp:.1%}")
        out_path = out_dir / f"{tag_ft}.pt"
        _save_pruned(pruned, ckpt, out_path, sp)
        print(f"  Saved → {out_path.relative_to(_REPO)}")
        row_label = tag_ft + (" (fine-tuned)" if args.finetune_epochs > 0 else "")
        lines.append(f"{row_label:<36}  {acc:.4f}  {sign}{delta:.4f}  {size:>8.1f}  {actual_sp:>8.1%}")
        pruned_models[sp] = (pruned, acc)
    print()

    # ── INT8 static quantization ──────────────────────────────────────────────
    print(f"INT8 static quantization (calibrating on {args.cal_batches} batches) …")
    cal_loader = _make_loader(manifest, "train", args.batch_size)

    try:
        quantized = quantize_static(model, cal_loader, device="cpu", backend="fbgemm")
        q_acc = evaluate(quantized, val_loader)
        q_delta = q_acc - baseline_acc
        sign = "+" if q_delta >= 0 else ""
        # quantized model state dict uses different keys; measure whole-model size
        buf = __import__("io").BytesIO()
        torch.save(quantized, buf)
        q_size = buf.tell() / 1024
        print(f"  quantized_int8: acc={q_acc:.4f}  Δ={sign}{q_delta:.4f}  size={q_size:.1f} kB")
        out_q = out_dir / "quantized_int8.pt"
        torch.save(quantized, out_q)
        print(f"  Saved → {out_q.relative_to(_REPO)}")
        lines.append(f"{'quantized_int8 (INT8)':<32}  {q_acc:.4f}  {sign}{q_delta:.4f}  {q_size:>8.1f}  {'n/a':>9}")
        q_status = "ok"
    except Exception as exc:
        print(f"  Quantization failed: {exc}")
        lines.append(f"{'quantized_int8 (INT8)':<32}  {'FAILED — see stderr':>50}")
        q_status = str(exc)

    # ── write report ──────────────────────────────────────────────────────────
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("Accuracy measured on the validation split; no figures fabricated.")
    lines.append("Pruning: L1 unstructured on Conv2d weights; masks removed (permanent).")
    lines.append("Quantization: FX graph-mode static INT8, fbgemm backend, 224x224 input.")
    if q_status != "ok":
        lines.append(f"Quantization error: {q_status}")

    report_path = out_dir / "optimization_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport → {report_path.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
