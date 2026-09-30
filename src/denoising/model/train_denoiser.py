"""Training loop for the DnCNN denoiser.

Separate from ``train.py``, which trains the :class:`~denoising.model.cnn.NoiseClassifierCNN`.
Nothing in this file touches the classifier checkpoint or its dataset.

Dataset
-------
:class:`DenoisingDataset` reads paired (noisy, clean) rows from the denoising
manifest (``data/denoising/manifest.csv``).  Each sample is a dict with keys
``noisy`` and ``clean`` — float32 tensors in [0, 1], shape (1, H, W).

Training
--------
:func:`train_denoiser` trains a :class:`~denoising.model.dncnn.DnCNN` with
MSELoss and the Adam optimiser.  The model is evaluated on the validation
split after every epoch.  Checkpoint is saved whenever the validation PSNR
improves.

Checkpoint format
-----------------
``models/checkpoints/denoiser.pt`` — a ``torch.save`` dict with keys
``model_state_dict``, ``epoch``, ``train_loss``, ``val_loss``, ``val_psnr``
and ``config`` (a plain-dict snapshot of the training config).  Metadata is
also written to ``models/metadata/denoiser_result.json``.
"""

from __future__ import annotations

import csv
import json
import math
import time

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..config import DenoisingTrainingConfig, load_denoising_training_config
from ..logging_utils import get_logger
from .dncnn import DnCNN, build_dncnn

__all__ = [
    "DenoisingDataset",
    "build_optimizer",
    "train_denoiser",
]

_LOG = get_logger(__name__)


class DenoisingDataset(Dataset):
    """Paired (noisy, clean) dataset for the DnCNN denoiser.

    Reads from the manifest produced by :func:`~denoising.dataset.denoising_generate.generate_denoising_dataset`.
    Returns dicts ``{"noisy": tensor, "clean": tensor}`` with float32 tensors
    in [0, 1], shape (1, H, W).

    Args:
        manifest:  Path to the CSV manifest.
        split:     "train", "val", or "test".
        root:      Optional root to prepend to relative paths in the manifest.
    """

    def __init__(
        self, manifest: Path | str, split: str, root: Path | str | None = None
    ) -> None:
        manifest = Path(manifest)
        self._root = Path(root) if root is not None else manifest.parent.parent
        self._rows: list[dict[str, str]] = []
        with manifest.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["split"] == split:
                    self._rows.append(row)
        if not self._rows:
            raise ValueError(
                f"no rows for split={split!r} in manifest {manifest}"
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self._rows[index]
        noisy = self._load(row["noisy_path"])
        clean = self._load(row["clean_path"])
        return {"noisy": noisy, "clean": clean}

    def _load(self, path_str: str) -> torch.Tensor:
        p = Path(path_str)
        if not p.is_absolute():
            p = self._root / p
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required for DenoisingDataset") from exc
        data = p.read_bytes()
        buf = __import__("numpy").frombuffer(data, dtype=__import__("numpy").uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read image at {p}")
        tensor = torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0)
        return tensor


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_optimizer(
    model: nn.Module, config: DenoisingTrainingConfig
) -> torch.optim.Optimizer:
    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"unknown optimizer: {name!r}")


def _psnr(mse: float) -> float:
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _validate(
    model: DnCNN, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            out = model(noisy)
            loss = criterion(out, clean)
            total_loss += loss.item() * noisy.size(0)
            count += noisy.size(0)
    val_loss = total_loss / max(count, 1)
    val_psnr = _psnr(val_loss)
    return val_loss, val_psnr


def _save_checkpoint(
    path: Path,
    model: DnCNN,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_psnr: float,
    config: DenoisingTrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "DnCNN",
            "depth": model.depth,
            "filters": model.filters,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "config": {
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "optimizer": config.optimizer,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def _write_metadata(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_psnr: float,
    elapsed_seconds: float,
    config: DenoisingTrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "model_type": "DnCNN",
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_psnr_db": val_psnr,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "config": {
            "depth": config.model.depth,
            "filters": config.model.filters,
            "input_channels": config.model.input_channels,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer,
        },
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def train_denoiser(
    manifest: Path | str,
    config: DenoisingTrainingConfig | None = None,
    *,
    dataset_root: Path | str | None = None,
    verbose: bool = True,
) -> DnCNN:
    """Train a DnCNN denoiser and save a checkpoint.

    Args:
        manifest:      Path to the denoising manifest CSV.
        config:        Training config; loads ``configs/denoising_training.yaml``
                       if ``None``.
        dataset_root:  Root directory prepended to relative manifest paths.
        verbose:       Print per-epoch progress.

    Returns:
        The best-performing :class:`DnCNN` (loaded from the saved checkpoint).
    """
    if config is None:
        config = load_denoising_training_config()

    manifest = Path(manifest)
    device = _resolve_device(config.device)

    if config.seed is not None:
        torch.manual_seed(config.seed)

    train_ds = DenoisingDataset(manifest, "train", root=dataset_root)
    val_ds = DenoisingDataset(manifest, "val", root=dataset_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_dncnn(
        depth=config.model.depth,
        filters=config.model.filters,
        input_channels=config.model.input_channels,
        device=device,
    )

    if config.resume and config.checkpoint_path.exists():
        ckpt = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("val_psnr", 0.0)
        _LOG.info("resumed from epoch %d, val_psnr=%.2f dB", start_epoch - 1, best_psnr)
    else:
        start_epoch = 0
        best_psnr = -float("inf")

    optimizer = build_optimizer(model, config)
    criterion = nn.MSELoss()

    patience_counter = 0
    best_train_loss = float("inf")
    t0 = time.monotonic()

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            optimizer.zero_grad()
            out = model(noisy)
            loss = criterion(out, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * noisy.size(0)

        train_loss = epoch_loss / max(len(train_ds), 1)
        val_loss, val_psnr = _validate(model, val_loader, criterion, device)

        if verbose:
            print(
                f"  epoch {epoch+1:3d}/{config.epochs} "
                f"train_loss={train_loss:.6f}  "
                f"val_loss={val_loss:.6f}  "
                f"val_psnr={val_psnr:.2f} dB"
            )

        if val_psnr > best_psnr + config.early_stopping.min_delta:
            best_psnr = val_psnr
            best_train_loss = train_loss
            patience_counter = 0
            _save_checkpoint(
                config.checkpoint_path,
                model,
                epoch,
                train_loss,
                val_loss,
                val_psnr,
                config,
            )
        else:
            patience_counter += 1
            if config.early_stopping.enabled and patience_counter >= config.early_stopping.patience:
                if verbose:
                    print(
                        f"  early stopping at epoch {epoch+1} "
                        f"(no improvement for {patience_counter} epochs)"
                    )
                break

    elapsed = time.monotonic() - t0
    _write_metadata(
        config.metadata_path,
        epoch,
        best_train_loss,
        0.0,
        best_psnr,
        elapsed,
        config,
    )

    # Load and return the best checkpoint
    best_ckpt = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    return model
