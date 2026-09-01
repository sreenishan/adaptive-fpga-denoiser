"""Training loop for the noise classifier (spec phase 7).

Usage::

    from denoising.config import load_training_config
    from denoising.model.train import train

    config = load_training_config()
    result = train(config, dataset_dir=Path("data/processed"))

The loop:
1. Loads the manifest from ``dataset_dir/manifest.csv``.
2. Builds weighted random sampling to counter the 1:3:3:3 class imbalance.
3. Trains with Adam/AdamW/SGD, optional LR scheduler, and early stopping.
4. Saves the best checkpoint (by validation accuracy) to ``checkpoint_dir``.
5. Returns a ``TrainResult`` with the per-epoch history.

Class imbalance:
    The dataset has one clean image per source but three noisy variants per
    class (one per intensity level), so the minority class (clean) is
    under-represented 3:1.  ``WeightedRandomSampler`` re-balances the training
    batches without duplicating files on disk.  The validation set is left
    unweighted so the accuracy metric reflects the real distribution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..config import CLASSES, TrainingConfig
from ..preprocessing import load_image, to_model_input
from .cnn import NoiseClassifierCNN, build_model, model_info

__all__ = ["train", "TrainResult", "NoiseDataset"]


# ─── Dataset ──────────────────────────────────────────────────────────────────


class NoiseDataset(Dataset):
    """Reads images listed in a manifest CSV produced by the dataset generator.

    Args:
        manifest: DataFrame with columns ``path``, ``label``, ``split``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        image_shape: ``(H, W)`` — images are resized to this before returning.
        augment: horizontal/vertical flip augmentation (training only).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        image_shape: tuple[int, int] = (224, 224),
        *,
        augment: bool = False,
    ) -> None:
        rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"no rows in manifest for split={split!r}")
        self._paths = rows["path"].tolist()
        self._labels = rows["label"].astype(int).tolist()
        self._h, self._w = image_shape
        self._augment = augment

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = load_image(Path(self._paths[idx]))
        # to_model_input resizes and normalises to float32 [0,1] shape (1,H,W)
        from ..config import ImageConfig

        tensor_np = to_model_input(img, ImageConfig(self._w, self._h, True), add_batch=False)
        x = torch.from_numpy(tensor_np)
        if self._augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])  # horizontal flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[1])  # vertical flip
        return x, self._labels[idx]

    @property
    def labels(self) -> list[int]:
        return list(self._labels)


# ─── Result ───────────────────────────────────────────────────────────────────


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    elapsed_s: float


@dataclass
class TrainResult:
    history: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_acc: float = 0.0
    checkpoint_path: Path | None = None
    metadata_path: Path | None = None
    stopped_early: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_val_acc": round(self.best_val_acc, 6),
            "stopped_early": self.stopped_early,
            "checkpoint_path": str(self.checkpoint_path),
            "history": [
                {
                    "epoch": m.epoch,
                    "train_loss": round(m.train_loss, 6),
                    "train_acc": round(m.train_acc, 6),
                    "val_loss": round(m.val_loss, 6),
                    "val_acc": round(m.val_acc, 6),
                    "lr": m.lr,
                    "elapsed_s": round(m.elapsed_s, 2),
                }
                for m in self.history
            ],
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _build_optimizer(
    model: nn.Module, config: TrainingConfig
) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9
        )
    raise ValueError(f"unknown optimizer {config.optimizer!r}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> Any | None:
    sched = config.scheduler
    epochs = config.epochs
    if sched is None:
        return None
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    if sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    raise ValueError(f"unknown scheduler {sched!r}")


def _class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Weight for each class = 1 / count, normalised to sum to num_classes."""
    counts = torch.zeros(num_classes)
    for lbl in labels:
        counts[lbl] += 1
    counts = counts.clamp(min=1)
    weights = num_classes / counts
    return weights / weights.sum() * num_classes


def _sample_weights(labels: list[int], class_weights: torch.Tensor) -> list[float]:
    return [float(class_weights[lbl]) for lbl in labels]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """One forward pass over *loader*.  If *optimizer* is None, no gradients."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total


# ─── Main entry ───────────────────────────────────────────────────────────────


def train(
    config: TrainingConfig,
    dataset_dir: Path,
    *,
    verbose: bool = True,
) -> TrainResult:
    """Train the noise classifier and return the result.

    Args:
        config: Loaded training configuration.
        dataset_dir: Directory containing ``manifest.csv`` and image files.
        verbose: Print per-epoch progress.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = _resolve_device(config.device)
    if verbose:
        print(f"Device: {device}")

    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {manifest_path}; "
            "run: denoising-generate-dataset --output <dataset_dir>"
        )
    manifest = pd.read_csv(manifest_path)

    # Infer image shape from the first row
    first_img = load_image(Path(manifest["path"].iloc[0]))
    h, w = first_img.shape

    train_ds = NoiseDataset(manifest, "train", (h, w), augment=True)
    val_ds = NoiseDataset(manifest, "val", (h, w), augment=False)

    if verbose:
        print(f"Train: {len(train_ds)} images   Val: {len(val_ds)} images")

    # Weighted sampler to counter 1:3:3:3 class imbalance
    cw = _class_weights(train_ds.labels, config.model.num_classes)
    sw = _sample_weights(train_ds.labels, cw)
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    bs = config.batch_size
    nw = config.num_workers
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    model = build_model(config).to(device)
    if verbose:
        print(model_info(model, (1, h, w)))

    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    # Cross-entropy with class weights applied to the loss (second defence
    # against imbalance, complementing the sampler)
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))

    ckpt_dir = Path(config.checkpoint_dir)
    meta_dir = Path(config.metadata_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    es = config.early_stopping
    patience_left = es.patience if es.enabled else None
    result = TrainResult(checkpoint_path=ckpt_path)

    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)
        elapsed = time.perf_counter() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        m = EpochMetrics(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed)
        result.history.append(m)

        if verbose:
            print(
                f"Epoch {epoch:3d}/{config.epochs}  "
                f"loss {train_loss:.4f}/{val_loss:.4f}  "
                f"acc {train_acc:.4f}/{val_acc:.4f}  "
                f"lr {current_lr:.2e}  {elapsed:.1f}s"
            )

        # Checkpoint on best val accuracy
        if val_acc > result.best_val_acc + (es.min_delta if es.enabled else 0.0):
            result.best_val_acc = val_acc
            result.best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "classes": list(CLASSES),
                    "model_config": {
                        "num_classes": config.model.num_classes,
                        "input_channels": config.model.input_channels,
                        "base_channels": config.model.base_channels,
                        "dropout": config.model.dropout,
                    },
                },
                ckpt_path,
            )
            if patience_left is not None:
                patience_left = es.patience
        else:
            if patience_left is not None:
                patience_left -= 1
                if patience_left <= 0:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (patience exhausted)")
                    result.stopped_early = True
                    break

        # LR scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    meta_path = meta_dir / "training_result.json"
    meta_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    result.metadata_path = meta_path

    if verbose:
        print(
            f"\nBest val accuracy: {result.best_val_acc:.4f} at epoch {result.best_epoch}"
        )
        print(f"Checkpoint: {ckpt_path}")
        print(f"Metadata:   {meta_path}")

    return result
