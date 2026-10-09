"""CLI: evaluate denoising strategies on the test split (spec phases 13–14).

Usage::

    python scripts/evaluate_denoising.py --dataset data/generated
    python scripts/evaluate_denoising.py --dataset data/generated --checkpoint models/checkpoints/best_model.pt
    python scripts/evaluate_denoising.py --dataset data/generated --output results/eval.json

Strategies compared
-------------------
1. no_filter    — identity (the floor every filter must beat)
2. fixed_median — always apply median, regardless of noise type
3. gt_adaptive  — use ground-truth label to pick the right filter (the ceiling)
4. cnn_adaptive — use the trained CNN to classify then pick the filter
                  (only when --checkpoint is supplied and the file exists)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate denoising strategies on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, type=Path,
                   help="Dataset directory containing manifest.csv.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to trained model checkpoint for cnn_adaptive strategy.")
    p.add_argument("--split", default="test",
                   help="Which split to evaluate on.")
    p.add_argument("--output", type=Path, default=None,
                   help="Save the full report as JSON to this path.")
    p.add_argument("--csv", type=Path, default=None,
                   help="Save per-image records as CSV to this path.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output.")
    return p.parse_args()


def _psnr_str(v: float) -> str:
    return "   ∞  " if math.isinf(v) or v > 1e14 else f"{v:6.2f}"


def _print_report(report) -> None:
    strats = report.strategies
    names = list(strats.keys())
    label_map = {
        "no_filter":    "No filter  ",
        "fixed_median": "Fixed median",
        "gt_adaptive":  "GT adaptive ",
        "cnn_adaptive": "CNN adaptive",
    }
    noise_label = {
        "salt_pepper": "Salt&Pepper",
        "gaussian":    "Gaussian   ",
        "speckle":     "Speckle    ",
    }

    print()
    print("=" * 72)
    print("  DENOISING EVALUATION REPORT")
    print(f"  Dataset : {report.dataset_dir}")
    print(f"  Split   : test   N = {report.n_test} noisy images")
    print("=" * 72)

    # ── Overall PSNR comparison ───────────────────────────────────────────
    print()
    print("  Overall (avg over all noisy classes)")
    print(f"  {'Strategy':<14}  {'MSE':>8}  {'PSNR (dB)':>10}  {'SSIM':>7}")
    print("  " + "-" * 44)
    baseline_psnr = strats["no_filter"].overall_psnr if "no_filter" in strats else 0.0
    for name in names:
        s = strats[name]
        delta = s.overall_psnr - baseline_psnr
        delta_str = f"  ({delta:+.2f})" if name != "no_filter" else ""
        print(
            f"  {label_map.get(name, name):<14}  "
            f"{s.overall_mse:8.2f}  "
            f"{_psnr_str(s.overall_psnr):>10}  "
            f"{s.overall_ssim:7.4f}"
            f"{delta_str}"
        )

    # ── Per-class breakdown ───────────────────────────────────────────────
    for cls in ("salt_pepper", "gaussian", "speckle"):
        print()
        print(f"  {noise_label[cls]}  (per-strategy, PSNR delta vs no_filter)")
        print(f"  {'Strategy':<14}  {'N':>3}  {'MSE':>8}  {'PSNR':>9}  {'SSIM':>7}  {'ΔPSNR':>8}")
        print("  " + "-" * 56)
        for name in names:
            s = strats[name]
            cm = next((c for c in s.per_class if c.noise_class == cls), None)
            if cm is None:
                continue
            print(
                f"  {label_map.get(name, name):<14}  "
                f"{cm.n:3d}  "
                f"{cm.mse:8.2f}  "
                f"{_psnr_str(cm.psnr):>9}  "
                f"{cm.ssim:7.4f}  "
                f"{cm.psnr_delta:+8.2f}"
            )

    # ── Improvement summary ───────────────────────────────────────────────
    if "gt_adaptive" in strats and "no_filter" in strats:
        gt = strats["gt_adaptive"]
        nf = strats["no_filter"]
        print()
        print("  GT adaptive improvement over no_filter")
        print(f"  {'Class':<14}  {'ΔMSE':>9}  {'ΔPSNR':>9}  {'ΔSSIM':>8}")
        print("  " + "-" * 46)
        for cm in gt.per_class:
            print(
                f"  {noise_label.get(cm.noise_class, cm.noise_class):<14}  "
                f"{cm.mse_delta:+9.2f}  "
                f"{cm.psnr_delta:+9.2f}  "
                f"{cm.ssim_delta:+8.4f}"
            )

    if "cnn_adaptive" in strats and "gt_adaptive" in strats:
        cnn = strats["cnn_adaptive"]
        gt = strats["gt_adaptive"]
        psnr_gap = gt.overall_psnr - cnn.overall_psnr
        print()
        print(
            f"  CNN vs GT gap  : {psnr_gap:+.2f} dB PSNR  "
            f"(0.00 = classifier perfect, negative = CNN beats GT boundary)"
        )

    print()
    print("=" * 72)


def main() -> None:
    args = _parse()

    from denoising.config import load_inference_config
    from denoising.evaluation import evaluate

    config = load_inference_config()

    # Try to load classifier if checkpoint supplied
    classifier = None
    if args.checkpoint is not None and args.checkpoint.exists():
        try:
            from denoising.model.inference import load_classifier
            classifier = load_classifier(args.checkpoint, config)
            print(f"Loaded classifier from {args.checkpoint}")
        except Exception as exc:
            print(f"Warning: could not load classifier ({exc}). Skipping cnn_adaptive.")
    elif args.checkpoint is None:
        # Auto-discover default checkpoint
        default_ckpt = _ROOT / config.model_path
        if default_ckpt.exists():
            try:
                from denoising.model.inference import load_classifier
                classifier = load_classifier(default_ckpt, config)
                print(f"Auto-loaded classifier from {default_ckpt}")
            except Exception:
                pass

    report = evaluate(
        args.dataset,
        config=config,
        classifier=classifier,
        split=args.split,
        verbose=not args.quiet,
    )

    _print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.as_dict(), indent=2), encoding="utf-8"
        )
        print(f"Report saved to {args.output}")

    if args.csv:
        import pandas as pd
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for strategy, s in report.strategies.items():
            for rec in s.records:
                rows.append({"strategy": strategy, **rec})
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"Records saved to {args.csv}")


if __name__ == "__main__":
    main()
