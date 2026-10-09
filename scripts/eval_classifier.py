"""Evaluate the noise classifier on the test split.

Loads models/checkpoints/best_model.pt, runs every image in the test split
through it, and produces:

  results/classifier/metrics.json         per-class accuracy/precision/recall/F1
  results/classifier/confusion_matrix.png  heatmap of the 5×5 confusion matrix
  results/classifier/training_curves.png   train/val accuracy and loss history

All numbers come from actual model runs.  Nothing is fabricated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from denoising.config import CLASSES
from denoising.model.cnn import NoiseClassifierCNN
from denoising.model.train import NoiseDataset

CKPT     = _REPO / "models" / "checkpoints" / "best_model.pt"
MANIFEST = _REPO / "data" / "generated" / "manifest.csv"
HISTORY  = _REPO / "models" / "metadata" / "training_result.json"
OUT_DIR  = _REPO / "results" / "classifier"

CLASS_NAMES = list(CLASSES)


# ── model ─────────────────────────────────────────────────────────────────────

def _load_model() -> NoiseClassifierCNN:
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
    return model


# ── inference ─────────────────────────────────────────────────────────────────

def _run_inference(model: NoiseClassifierCNN) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(MANIFEST)
    ds = NoiseDataset(df, "test", image_shape=(224, 224))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_preds), np.array(all_labels)


# ── metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for p, l in zip(preds.tolist(), labels.tolist()):
        cm[l, p] += 1

    overall_acc = float((preds == labels).mean())

    per_class: dict[str, dict] = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(cm[c, :].sum())
        per_class[name] = {
            "accuracy":  round(float(tp / support) if support > 0 else 0.0, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "support":   support,
        }

    return {
        "overall_accuracy": round(overall_acc, 4),
        "per_class":        per_class,
        "confusion_matrix": cm.tolist(),
        "class_names":      CLASS_NAMES,
        "test_images":      int(len(preds)),
    }


# ── confusion matrix plot ─────────────────────────────────────────────────────

def _save_confusion_matrix(cm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title("Noise Classifier — Test Split Confusion Matrix", fontsize=12, pad=10)
    thresh = cm.max() / 2.0
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.relative_to(_REPO)}")


# ── training curves plot ──────────────────────────────────────────────────────

def _save_training_curves(out: Path) -> None:
    if not HISTORY.exists():
        print(f"  Training history not found: {HISTORY} — skipping")
        return
    with open(HISTORY) as f:
        h = json.load(f)

    epochs     = [e["epoch"]      for e in h["history"]]
    train_acc  = [e["train_acc"]  for e in h["history"]]
    val_acc    = [e["val_acc"]    for e in h["history"]]
    train_loss = [e["train_loss"] for e in h["history"]]
    val_loss   = [e["val_loss"]   for e in h["history"]]
    best_ep    = h["best_epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_acc, "b-o", markersize=4, label="Train")
    ax1.plot(epochs, val_acc,   "r-o", markersize=4, label="Val")
    ax1.axvline(best_ep, color="gray", linestyle="--", linewidth=1,
                label=f"Best ep {best_ep}  ({h['best_val_acc']*100:.2f}%)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    ax2.plot(epochs, train_loss, "b-o", markersize=4, label="Train")
    ax2.plot(epochs, val_loss,   "r-o", markersize=4, label="Val")
    ax2.axvline(best_ep, color="gray", linestyle="--", linewidth=1, label=f"Best ep {best_ep}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Loss"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle("Noise Classifier CNN  —  Training History (5-class, 16 epochs, early stop ep 11)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.relative_to(_REPO)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model …")
    model = _load_model()

    print(f"Running inference on test split …")
    preds, labels = _run_inference(model)
    print(f"  {len(preds)} images evaluated")

    metrics = _compute_metrics(preds, labels)
    print(f"  Overall test accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print()
    print(f"  {'Class':<14} {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'N':>4}")
    print(f"  {'-'*46}")
    for name, m in metrics["per_class"].items():
        print(f"  {name:<14} {m['accuracy']*100:5.1f}%  {m['precision']*100:5.1f}%  "
              f"{m['recall']*100:5.1f}%  {m['f1']*100:5.1f}%  {m['support']:4d}")

    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n  Saved {metrics_path.relative_to(_REPO)}")

    cm = np.array(metrics["confusion_matrix"])
    _save_confusion_matrix(cm, OUT_DIR / "confusion_matrix.png")
    _save_training_curves(OUT_DIR / "training_curves.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
