"""Denoising evaluation engine (spec phases 13–14).

Runs four strategies on the test split and returns structured results:

1. **no_filter**   — identity; output equals input (the floor).
2. **fixed_median** — always apply the 3×3 median regardless of noise type.
3. **gt_adaptive**  — select the filter from the ground-truth label; this is
   the ceiling the classifier is trying to reach.
4. **cnn_adaptive** — select the filter using the trained CNN; present only
   when a classifier is supplied.

For each strategy the results are broken down per noise class, with MSE, PSNR,
SSIM, and the delta versus the no-filter baseline.

The clean class is excluded from the noisy-class metrics because there is
nothing to improve (PSNR is infinite at perfect input), but it is included in
the raw records so callers can inspect bypass behaviour on clean images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import CLASSES, InferenceConfig, load_inference_config
from ..filters import apply_filter
from ..filters.selector import FILTER_FOR_CLASS
from ..metrics.image_quality import calculate_quality
from ..preprocessing import load_image

__all__ = [
    "StrategyResult",
    "ClassMetrics",
    "EvalReport",
    "evaluate",
]

STRATEGIES = ("no_filter", "fixed_median", "gt_adaptive", "cnn_adaptive")
NOISY_CLASSES = ("salt_pepper", "gaussian", "speckle")


# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass
class ClassMetrics:
    noise_class: str
    n: int
    mse: float
    psnr: float
    ssim: float
    mse_delta: float = 0.0   # vs no_filter
    psnr_delta: float = 0.0
    ssim_delta: float = 0.0


@dataclass
class StrategyResult:
    strategy: str
    n_total: int
    overall_mse: float
    overall_psnr: float
    overall_ssim: float
    per_class: list[ClassMetrics] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalReport:
    strategies: dict[str, StrategyResult] = field(default_factory=dict)
    n_test: int = 0
    dataset_dir: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_test": self.n_test,
            "dataset_dir": self.dataset_dir,
            "strategies": {
                name: {
                    "overall": {
                        "mse": round(s.overall_mse, 4),
                        "psnr": round(s.overall_psnr, 4),
                        "ssim": round(s.overall_ssim, 4),
                    },
                    "per_class": [
                        {
                            "class": c.noise_class,
                            "n": c.n,
                            "mse": round(c.mse, 4),
                            "psnr": round(c.psnr, 4),
                            "ssim": round(c.ssim, 4),
                            "mse_delta": round(c.mse_delta, 4),
                            "psnr_delta": round(c.psnr_delta, 4),
                            "ssim_delta": round(c.ssim_delta, 4),
                        }
                        for c in s.per_class
                    ],
                }
                for name, s in self.strategies.items()
            },
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _apply_strategy(
    noisy: np.ndarray,
    noise_class: str,
    strategy: str,
    config: InferenceConfig,
    classifier: Any | None,
) -> tuple[np.ndarray, str]:
    """Return (output_image, filter_used)."""
    filters_cfg = config.filters

    if strategy == "no_filter":
        return noisy.copy(), "bypass"

    if strategy == "fixed_median":
        out = apply_filter(noisy, "median", kernel_size=filters_cfg.median.kernel_size)
        return out, "median"

    if strategy == "gt_adaptive":
        filt = FILTER_FOR_CLASS[noise_class]
        params: dict[str, Any] = {}
        if filt == "median":
            params["kernel_size"] = filters_cfg.median.kernel_size
        elif filt == "gaussian":
            params["kernel_size"] = filters_cfg.gaussian.kernel_size
            params["integer_kernel"] = True
        elif filt == "wiener":
            params["kernel_size"] = filters_cfg.wiener.kernel_size
            params["noise_variance"] = filters_cfg.wiener.noise_variance
        return apply_filter(noisy, filt, **params), filt

    if strategy == "cnn_adaptive":
        if classifier is None:
            raise ValueError("cnn_adaptive requires a classifier")
        predicted_class, _ = classifier.predict(noisy)
        filt = FILTER_FOR_CLASS[predicted_class]
        params = {}
        if filt == "median":
            params["kernel_size"] = filters_cfg.median.kernel_size
        elif filt == "gaussian":
            params["kernel_size"] = filters_cfg.gaussian.kernel_size
            params["integer_kernel"] = True
        elif filt == "wiener":
            params["kernel_size"] = filters_cfg.wiener.kernel_size
            params["noise_variance"] = filters_cfg.wiener.noise_variance
        return apply_filter(noisy, filt, **params), filt

    raise ValueError(f"unknown strategy: {strategy!r}")


def _mean_finite(values: list[float]) -> float:
    finite = [v for v in values if not (v != v) and v < 1e15]  # exclude nan/inf
    return float(np.mean(finite)) if finite else float("nan")


# ─── Core evaluation ──────────────────────────────────────────────────────────


def evaluate(
    dataset_dir: str | Path,
    config: InferenceConfig | None = None,
    classifier: Any | None = None,
    *,
    split: str = "test",
    verbose: bool = True,
) -> EvalReport:
    """Run all strategies on *split* and return an :class:`EvalReport`.

    Args:
        dataset_dir: Directory containing ``manifest.csv``.
        config: Inference config; loaded from default path if None.
        classifier: An object implementing ``predict(image) -> (str, float)``.
            When None, the ``cnn_adaptive`` strategy is skipped.
        split: Dataset split to evaluate on (``"test"`` by default).
        verbose: Print progress.
    """
    if config is None:
        config = load_inference_config()

    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    label_to_class = {i: cls for i, cls in enumerate(CLASSES)}

    # Build a lookup: source_id → clean image path
    clean_rows = manifest[(manifest["split"] == split) & (manifest["noise_type"] == "clean")]
    clean_lookup: dict[str, Path] = {
        row["source_id"]: Path(row["path"])
        for _, row in clean_rows.iterrows()
    }

    # Evaluate on noisy test images only (skip clean — PSNR is infinite)
    test_rows = manifest[
        (manifest["split"] == split) & (manifest["noise_type"] != "clean")
    ].reset_index(drop=True)

    n_test = len(test_rows)
    if verbose:
        print(f"Evaluating {n_test} {split} images …")

    active_strategies = list(STRATEGIES)
    if classifier is None:
        active_strategies = [s for s in active_strategies if s != "cnn_adaptive"]

    # Per-strategy, per-image records
    raw: dict[str, list[dict[str, Any]]] = {s: [] for s in active_strategies}

    for idx, row in test_rows.iterrows():
        noise_class = label_to_class[int(row["label"])]
        source_id = row["source_id"]
        noisy_path = Path(row["path"])

        if source_id not in clean_lookup:
            continue  # no clean reference in this split — skip

        noisy = load_image(noisy_path)
        clean = load_image(clean_lookup[source_id])

        if noisy.shape != clean.shape:
            continue

        for strategy in active_strategies:
            try:
                output, filt_used = _apply_strategy(noisy, noise_class, strategy, config, classifier)
                q = calculate_quality(clean, output)
                raw[strategy].append({
                    "source_id": source_id,
                    "noise_class": noise_class,
                    "noise_parameter": row.get("noise_parameter", None),
                    "filter_used": filt_used,
                    "mse": q.mse,
                    "psnr": q.psnr,
                    "ssim": q.ssim,
                })
            except Exception as exc:
                if verbose:
                    print(f"  [{strategy}] {noisy_path.name}: {exc}")

        if verbose and (int(idx) + 1) % 10 == 0:
            print(f"  {int(idx) + 1}/{n_test} done")

    # Build baseline (no_filter) lookup for deltas
    baseline_by_key: dict[tuple[str, str], dict[str, float]] = {}
    for rec in raw.get("no_filter", []):
        key = (rec["source_id"], rec["noise_class"])
        baseline_by_key[key] = {"mse": rec["mse"], "psnr": rec["psnr"], "ssim": rec["ssim"]}

    # Aggregate into StrategyResult
    results: dict[str, StrategyResult] = {}
    for strategy in active_strategies:
        records = raw[strategy]
        if not records:
            continue

        all_mse = [r["mse"] for r in records]
        all_psnr = [r["psnr"] for r in records if r["psnr"] < 1e15]
        all_ssim = [r["ssim"] for r in records]

        per_class: list[ClassMetrics] = []
        for cls in NOISY_CLASSES:
            cls_recs = [r for r in records if r["noise_class"] == cls]
            if not cls_recs:
                continue
            mse_vals = [r["mse"] for r in cls_recs]
            psnr_vals = [r["psnr"] for r in cls_recs if r["psnr"] < 1e15]
            ssim_vals = [r["ssim"] for r in cls_recs]

            avg_mse = float(np.mean(mse_vals))
            avg_psnr = _mean_finite(psnr_vals)
            avg_ssim = float(np.mean(ssim_vals))

            # Delta vs no_filter for same class
            base_recs = [r for r in raw.get("no_filter", []) if r["noise_class"] == cls]
            base_mse = float(np.mean([r["mse"] for r in base_recs])) if base_recs else avg_mse
            base_psnr = _mean_finite([r["psnr"] for r in base_recs if r["psnr"] < 1e15]) if base_recs else avg_psnr
            base_ssim = float(np.mean([r["ssim"] for r in base_recs])) if base_recs else avg_ssim

            per_class.append(ClassMetrics(
                noise_class=cls,
                n=len(cls_recs),
                mse=avg_mse,
                psnr=avg_psnr,
                ssim=avg_ssim,
                mse_delta=avg_mse - base_mse,
                psnr_delta=avg_psnr - base_psnr,
                ssim_delta=avg_ssim - base_ssim,
            ))

        results[strategy] = StrategyResult(
            strategy=strategy,
            n_total=len(records),
            overall_mse=float(np.mean(all_mse)) if all_mse else float("nan"),
            overall_psnr=_mean_finite(all_psnr),
            overall_ssim=float(np.mean(all_ssim)) if all_ssim else float("nan"),
            per_class=per_class,
            records=records,
        )

    return EvalReport(
        strategies=results,
        n_test=n_test,
        dataset_dir=str(dataset_dir),
    )
