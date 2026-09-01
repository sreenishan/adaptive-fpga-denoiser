"""Evaluation engine tests (spec phases 13–14)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from denoising.config import CLASSES, load_inference_config
from denoising.evaluation import EvalReport, evaluate


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def small_dataset(tmp_path: Path):
    """Minimal test-split dataset: 2 sources × 4 classes, all in 'test'."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    label_map = {cls: i for i, cls in enumerate(CLASSES)}
    noise_amounts = {"salt_pepper": 0.25, "gaussian": 0.15, "speckle": 0.15}
    rows = []

    for src_idx in range(2):
        src_id = f"src_{src_idx:03d}"
        # Structured gradient — filters improve noisy structured images.
        # Random-pixel images have high intrinsic "noise" so filters make them
        # worse by design; that is not what these tests are measuring.
        yy, xx = np.mgrid[0:64, 0:64]
        clean_img = ((xx * 3 + yy * 2 + src_idx * 17) % 200 + 28).astype(np.uint8)
        clean_img[16:32, 16:32] = 200  # bright block

        # Clean image
        clean_path = img_dir / f"{src_id}_clean.png"
        cv2.imwrite(str(clean_path), clean_img)
        rows.append({
            "path": str(clean_path), "split": "test", "label": 0,
            "source_id": src_id, "noise_type": "clean",
            "noise_parameter": 0.0, "seed": src_idx,
        })

        # Noisy variants
        for noise_type, amount in noise_amounts.items():
            from denoising.noise import (
                add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise,
            )
            if noise_type == "salt_pepper":
                noisy = add_salt_pepper_noise(clean_img, amount, seed=src_idx)
            elif noise_type == "gaussian":
                noisy = add_gaussian_noise(clean_img, sigma=amount, seed=src_idx)
            else:
                noisy = add_speckle_noise(clean_img, variance=amount, seed=src_idx)

            noisy_path = img_dir / f"{src_id}_{noise_type}.png"
            cv2.imwrite(str(noisy_path), noisy)
            rows.append({
                "path": str(noisy_path), "split": "test",
                "label": label_map[noise_type],
                "source_id": src_id, "noise_type": noise_type,
                "noise_parameter": amount, "seed": src_idx,
            })

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return tmp_path


@pytest.fixture
def inference():
    return load_inference_config()


# ─── Basic structure ──────────────────────────────────────────────────────────


def test_evaluate_returns_report(small_dataset, inference):
    report = evaluate(small_dataset, config=inference, verbose=False)
    assert isinstance(report, EvalReport)
    assert report.n_test > 0


def test_report_has_three_strategies_without_classifier(small_dataset, inference):
    report = evaluate(small_dataset, config=inference, verbose=False)
    assert set(report.strategies.keys()) == {"no_filter", "fixed_median", "gt_adaptive"}
    assert "cnn_adaptive" not in report.strategies


def test_report_has_four_strategies_with_classifier(small_dataset, inference):
    class _StubClassifier:
        def predict(self, image):
            return "gaussian", 0.95

    report = evaluate(
        small_dataset, config=inference,
        classifier=_StubClassifier(), verbose=False,
    )
    assert "cnn_adaptive" in report.strategies


def test_per_class_covers_three_noisy_classes(small_dataset, inference):
    report = evaluate(small_dataset, config=inference, verbose=False)
    gt = report.strategies["gt_adaptive"]
    classes = {c.noise_class for c in gt.per_class}
    assert classes == {"salt_pepper", "gaussian", "speckle"}


def test_n_counts_are_positive(small_dataset, inference):
    report = evaluate(small_dataset, config=inference, verbose=False)
    for s in report.strategies.values():
        assert s.n_total > 0
        for c in s.per_class:
            assert c.n > 0


def test_missing_manifest_raises(tmp_path, inference):
    with pytest.raises(FileNotFoundError, match="manifest"):
        evaluate(tmp_path, config=inference, verbose=False)


# ─── Metric ordering ──────────────────────────────────────────────────────────


def test_gt_adaptive_psnr_exceeds_no_filter(small_dataset, inference):
    """The correct filter must always improve PSNR over doing nothing."""
    report = evaluate(small_dataset, config=inference, verbose=False)
    nf = report.strategies["no_filter"]
    gt = report.strategies["gt_adaptive"]
    assert gt.overall_psnr > nf.overall_psnr


def test_gt_adaptive_mse_below_no_filter(small_dataset, inference):
    report = evaluate(small_dataset, config=inference, verbose=False)
    nf = report.strategies["no_filter"]
    gt = report.strategies["gt_adaptive"]
    assert gt.overall_mse < nf.overall_mse


def test_gt_adaptive_psnr_delta_positive_per_class(small_dataset, inference):
    """Per-class PSNR delta (vs no_filter) must be positive for gt_adaptive."""
    report = evaluate(small_dataset, config=inference, verbose=False)
    gt = report.strategies["gt_adaptive"]
    for cm in gt.per_class:
        assert cm.psnr_delta > 0, f"psnr_delta negative for {cm.noise_class}"


def test_median_beats_no_filter_on_salt_pepper(small_dataset, inference):
    report = evaluate(small_dataset, config=inference, verbose=False)
    nf = report.strategies["no_filter"]
    fm = report.strategies["fixed_median"]
    nf_sp = next(c for c in nf.per_class if c.noise_class == "salt_pepper")
    fm_sp = next(c for c in fm.per_class if c.noise_class == "salt_pepper")
    assert fm_sp.psnr > nf_sp.psnr


# ─── as_dict ──────────────────────────────────────────────────────────────────


def test_as_dict_is_json_serialisable(small_dataset, inference):
    import json
    report = evaluate(small_dataset, config=inference, verbose=False)
    payload = report.as_dict()
    text = json.dumps(payload)
    assert "gt_adaptive" in text
    assert "psnr" in text


def test_as_dict_has_overall_and_per_class(small_dataset, inference):
    report = evaluate(small_dataset, config=inference, verbose=False)
    d = report.as_dict()
    for name in d["strategies"]:
        assert "overall" in d["strategies"][name]
        assert "per_class" in d["strategies"][name]
