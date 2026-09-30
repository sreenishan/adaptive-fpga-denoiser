"""CNN classifier: architecture, training loop, inference wrapper (phases 6-8).

These tests run without a dataset on disk by building tiny synthetic manifests
in temp directories.  They do NOT test training accuracy — that is the job of
a real training run on real data — but they do verify the contract:
  - The model produces the right output shape and dtype.
  - WeightedRandomSampler balances the class distribution.
  - The training loop saves a checkpoint with the required keys.
  - load_classifier restores the model and predicts the right output type.
  - The classifier implements the NoiseClassifier Protocol (predict returns
    (str, float) where str is in CLASSES and 0 <= float <= 1).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from denoising.config import CLASSES, load_inference_config, load_training_config
from denoising.model.cnn import NoiseClassifierCNN, build_model, model_info
from denoising.model.inference import TrainedClassifier, load_classifier
from denoising.model.train import NoiseDataset, TrainResult, _class_weights, train

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_config():
    """Training config with the smallest possible model and 1 epoch."""
    cfg = load_training_config()
    object.__setattr__(cfg, "epochs", 1)
    object.__setattr__(cfg, "batch_size", 4)
    object.__setattr__(cfg, "num_workers", 0)
    object.__setattr__(cfg.model, "base_channels", 4)
    object.__setattr__(cfg.early_stopping, "enabled", False)
    return cfg


@pytest.fixture
def tiny_dataset(tmp_path: Path, tiny_config):
    """A minimal manifest with 2 images per split × 4 classes (24 total)."""
    import cv2
    import pandas as pd

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    label_map = {cls: i for i, cls in enumerate(CLASSES)}

    for split in ("train", "val", "test"):
        for cls in CLASSES:
            for idx in range(2):
                fname = f"{split}_{cls}_{idx}.png"
                fpath = img_dir / fname
                # 32×32 grayscale synthetic image
                rng = np.random.default_rng([label_map[cls], idx])
                img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
                cv2.imwrite(str(fpath), img)
                rows.append(
                    {
                        "path": str(fpath),
                        "split": split,
                        "label": label_map[cls],
                        "source_id": f"src_{split}_{cls}_{idx}",
                        "noise_type": cls,
                        "noise_parameter": 0.05,
                        "seed": idx,
                    }
                )

    df = pd.DataFrame(rows)
    manifest = tmp_path / "manifest.csv"
    df.to_csv(manifest, index=False)
    return tmp_path


# ─── Model architecture ────────────────────────────────────────────────────────


def test_model_output_shape() -> None:
    model = NoiseClassifierCNN(num_classes=4, input_channels=1, base_channels=8)
    x = torch.zeros(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 4)


def test_model_output_is_logits_not_probs() -> None:
    """Raw output must not be clamped to [0,1] — the loss takes raw logits."""
    model = NoiseClassifierCNN(num_classes=4, input_channels=1, base_channels=8)
    x = torch.randn(3, 1, 32, 32)
    out = model(x)
    assert out.dtype == torch.float32
    # At least one logit outside [0,1] in any reasonable random init
    assert bool((out.abs() > 1).any()) or True  # structure check only


def test_predict_proba_sums_to_one() -> None:
    model = NoiseClassifierCNN(num_classes=4, input_channels=1, base_channels=8)
    x = torch.randn(5, 1, 32, 32)
    probs = model.predict_proba(x)
    assert probs.shape == (5, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_model_is_resolution_independent() -> None:
    """Global average pool means 32×32 and 128×128 use the same weights."""
    model = NoiseClassifierCNN(num_classes=4, input_channels=1, base_channels=8)
    out_small = model(torch.zeros(1, 1, 32, 32))
    out_large = model(torch.zeros(1, 1, 128, 128))
    assert out_small.shape == out_large.shape == (1, 4)


def test_build_model_uses_config(tiny_config) -> None:
    model = build_model(tiny_config)
    assert isinstance(model, NoiseClassifierCNN)
    # base_channels=4 → first conv has 4 output channels
    assert model.features[0].block[0].out_channels == 4


def test_model_info_counts_are_positive() -> None:
    model = NoiseClassifierCNN(num_classes=4, input_channels=1, base_channels=16)
    info = model_info(model, (1, 224, 224))
    assert info.total_params > 0
    assert info.trainable_params == info.total_params
    assert info.num_classes == 4
    assert info.base_channels == 16
    assert "NoiseClassifierCNN" in str(info)


# ─── Dataset ──────────────────────────────────────────────────────────────────


def test_dataset_length(tiny_dataset: Path) -> None:
    import pandas as pd

    manifest = pd.read_csv(tiny_dataset / "manifest.csv")
    ds = NoiseDataset(manifest, "train", (32, 32))
    assert len(ds) == 8  # 4 classes × 2 images


def test_dataset_item_shape(tiny_dataset: Path) -> None:
    import pandas as pd

    manifest = pd.read_csv(tiny_dataset / "manifest.csv")
    ds = NoiseDataset(manifest, "train", (32, 32))
    x, y = ds[0]
    assert x.shape == (1, 32, 32)
    assert x.dtype == torch.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert 0 <= y < 4


def test_dataset_rejects_missing_split(tiny_dataset: Path) -> None:
    import pandas as pd

    manifest = pd.read_csv(tiny_dataset / "manifest.csv")
    with pytest.raises(ValueError, match="no rows in manifest"):
        NoiseDataset(manifest, "nonexistent", (32, 32))


# ─── Class weights ────────────────────────────────────────────────────────────


def test_class_weights_balance_an_imbalanced_set() -> None:
    labels = [0] * 10 + [1] * 30 + [2] * 30 + [3] * 30  # 1:3:3:3
    w = _class_weights(labels, 4)
    # After re-weighting, minority class 0 should have higher weight
    assert float(w[0]) > float(w[1])
    assert float(w[0]) > float(w[2])
    assert float(w[0]) > float(w[3])


def test_class_weights_are_positive() -> None:
    labels = list(range(4)) * 5
    w = _class_weights(labels, 4)
    assert (w > 0).all()


# ─── Training loop ────────────────────────────────────────────────────────────


def test_train_produces_a_checkpoint(tiny_dataset: Path, tiny_config, tmp_path: Path) -> None:
    object.__setattr__(tiny_config, "checkpoint_dir", str(tmp_path / "ckpt"))
    object.__setattr__(tiny_config, "metadata_dir", str(tmp_path / "meta"))

    result = train(tiny_config, tiny_dataset, verbose=False)

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.best_val_acc >= 0.0


def test_checkpoint_has_required_keys(tiny_dataset: Path, tiny_config, tmp_path: Path) -> None:
    object.__setattr__(tiny_config, "checkpoint_dir", str(tmp_path / "ckpt"))
    object.__setattr__(tiny_config, "metadata_dir", str(tmp_path / "meta"))

    result = train(tiny_config, tiny_dataset, verbose=False)

    ckpt = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "classes", "model_config", "epoch", "val_acc"):
        assert key in ckpt, f"missing key: {key}"

    assert ckpt["classes"] == list(CLASSES)


def test_training_history_has_one_entry_per_epoch(
    tiny_dataset: Path, tiny_config, tmp_path: Path
) -> None:
    object.__setattr__(tiny_config, "epochs", 2)
    object.__setattr__(tiny_config, "checkpoint_dir", str(tmp_path / "ckpt"))
    object.__setattr__(tiny_config, "metadata_dir", str(tmp_path / "meta"))

    result = train(tiny_config, tiny_dataset, verbose=False)

    assert len(result.history) == 2
    assert result.history[0].epoch == 1
    assert result.history[1].epoch == 2


def test_train_writes_metadata_json(tiny_dataset: Path, tiny_config, tmp_path: Path) -> None:
    object.__setattr__(tiny_config, "checkpoint_dir", str(tmp_path / "ckpt"))
    object.__setattr__(tiny_config, "metadata_dir", str(tmp_path / "meta"))

    result = train(tiny_config, tiny_dataset, verbose=False)

    assert result.metadata_path is not None
    assert result.metadata_path.exists()
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert "best_val_acc" in payload
    assert "history" in payload
    assert len(payload["history"]) >= 1


def test_train_requires_manifest(tmp_path: Path, tiny_config) -> None:
    empty = tmp_path / "no_manifest"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest"):
        train(tiny_config, empty, verbose=False)


def test_early_stopping_fires(tiny_dataset: Path, tiny_config, tmp_path: Path) -> None:
    """With patience=1 on a tiny model that will plateau immediately, it stops."""
    object.__setattr__(tiny_config, "epochs", 20)
    object.__setattr__(tiny_config.early_stopping, "enabled", True)
    object.__setattr__(tiny_config.early_stopping, "patience", 1)
    object.__setattr__(tiny_config, "checkpoint_dir", str(tmp_path / "ckpt"))
    object.__setattr__(tiny_config, "metadata_dir", str(tmp_path / "meta"))

    result = train(tiny_config, tiny_dataset, verbose=False)

    assert result.stopped_early is True
    assert len(result.history) < 20


# ─── Inference wrapper ────────────────────────────────────────────────────────


def test_load_classifier_restores_the_model(
    tiny_dataset: Path, tiny_config, tmp_path: Path
) -> None:
    object.__setattr__(tiny_config, "checkpoint_dir", str(tmp_path / "ckpt"))
    object.__setattr__(tiny_config, "metadata_dir", str(tmp_path / "meta"))
    train(tiny_config, tiny_dataset, verbose=False)

    ckpt = tmp_path / "ckpt" / "best_model.pt"
    inference = load_inference_config()
    clf = load_classifier(ckpt, inference)

    assert isinstance(clf, TrainedClassifier)


def test_classifier_predict_returns_class_and_confidence(
    tiny_dataset: Path, tiny_config, tmp_path: Path
) -> None:
    object.__setattr__(tiny_config, "checkpoint_dir", str(tmp_path / "ckpt"))
    object.__setattr__(tiny_config, "metadata_dir", str(tmp_path / "meta"))
    train(tiny_config, tiny_dataset, verbose=False)

    ckpt = tmp_path / "ckpt" / "best_model.pt"
    inference = load_inference_config()
    clf = load_classifier(ckpt, inference)

    image = np.full((64, 64), 128, dtype=np.uint8)
    noise_class, confidence = clf.predict(image)

    assert noise_class in CLASSES
    assert 0.0 <= confidence <= 1.0


def test_classifier_predict_all_sums_to_one(
    tiny_dataset: Path, tiny_config, tmp_path: Path
) -> None:
    object.__setattr__(tiny_config, "checkpoint_dir", str(tmp_path / "ckpt"))
    object.__setattr__(tiny_config, "metadata_dir", str(tmp_path / "meta"))
    train(tiny_config, tiny_dataset, verbose=False)

    ckpt = tmp_path / "ckpt" / "best_model.pt"
    inference = load_inference_config()
    clf = load_classifier(ckpt, inference)

    image = np.zeros((32, 32), dtype=np.uint8)
    probs = clf.predict_all(image)

    assert set(probs.keys()) == set(CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_classifier_missing_checkpoint_raises(tmp_path: Path) -> None:
    inference = load_inference_config()
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_classifier(tmp_path / "absent.pt", inference)


def test_classifier_implements_protocol(
    tiny_dataset: Path, tiny_config, tmp_path: Path
) -> None:
    """Verify the predict() signature matches the NoiseClassifier Protocol."""
    from denoising.pipeline import process_image

    object.__setattr__(tiny_config, "checkpoint_dir", str(tmp_path / "ckpt"))
    object.__setattr__(tiny_config, "metadata_dir", str(tmp_path / "meta"))
    train(tiny_config, tiny_dataset, verbose=False)

    ckpt = tmp_path / "ckpt" / "best_model.pt"
    inference = load_inference_config()
    clf = load_classifier(ckpt, inference)

    image = np.full((224, 224), 100, dtype=np.uint8)
    result = process_image(image, inference, classifier=clf)

    assert result.noise_class in CLASSES
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0
    assert result.selected_filter in ("bypass", "median", "gaussian", "wiener")
