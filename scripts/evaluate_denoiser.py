#!/usr/bin/env python3
"""CLI: evaluate denoising strategies on the paired dataset.

Usage
-----
    python scripts/evaluate_denoiser.py [--dataset PATH] [--checkpoint PATH]
                                        [--split test] [--json OUT]

Runs four strategies on the specified split and prints per-level comparison
tables for each noise type.  The ``cnn_denoiser`` strategy requires a saved
checkpoint (``models/checkpoints/denoiser.pt``); if the checkpoint does not
exist it is automatically skipped with a warning.

Strategies
----------
no_filter     — unmodified noisy image (the floor)
fixed_median  — 3×3 median regardless of noise type
gt_adaptive   — ground-truth best filter per noise type (the ceiling)
cnn_denoiser  — DnCNN residual learning
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from denoising.config import load_inference_config
from denoising.evaluation.denoising_eval import (
    DenoisingEvalReport,
    evaluate_denoiser,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate denoising strategies on the paired 4-level dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default=str(_REPO / "data" / "denoising"),
        metavar="PATH",
        help="root of the denoising dataset (default: data/denoising/)",
    )
    p.add_argument(
        "--checkpoint",
        default=str(_REPO / "models" / "checkpoints" / "denoiser.pt"),
        metavar="PATH",
        help="DnCNN checkpoint (default: models/checkpoints/denoiser.pt)",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="dataset split to evaluate (default: test)",
    )
    p.add_argument(
        "--json",
        default=None,
        metavar="OUT",
        dest="json_out",
        help="write JSON report to this path",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress output",
    )
    return p.parse_args()


def _print_level_table(report: DenoisingEvalReport, strategy: str) -> None:
    levels = report.per_level.get(strategy, [])
    if not levels:
        print(f"  (no per-level data for {strategy})")
        return
    print(f"\n  {strategy} — per-level breakdown")
    print(f"  {'noise_type':<14} {'level':>5} {'n':>4}  {'MSE':>8}  {'PSNR':>8}  {'SSIM':>6}  {'ΔPSNR':>7}")
    print(f"  {'-'*14} {'-'*5} {'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}")
    for lv in sorted(levels, key=lambda x: (x.noise_type, x.noise_level)):
        psnr_str = f"{lv.psnr:8.2f}" if lv.psnr < 1e14 else "     inf"
        delta_str = f"{lv.psnr_delta:+7.2f}" if abs(lv.psnr_delta) < 1e14 else "    n/a"
        print(
            f"  {lv.noise_type:<14} {lv.noise_level:5d} {lv.n:4d}  "
            f"{lv.mse:8.5f}  {psnr_str}  {lv.ssim:6.4f}  {delta_str}"
        )


def main() -> int:
    args = _parse()

    config = load_inference_config()
    checkpoint = Path(args.checkpoint)

    inferencer = None
    if checkpoint.exists():
        try:
            from denoising.model.inference_denoiser import DnCNNInferencer
            inferencer = DnCNNInferencer.from_checkpoint(checkpoint)
            if not args.quiet:
                print(f"loaded checkpoint: {checkpoint}")
        except Exception as exc:
            print(f"warning: could not load checkpoint ({exc}); skipping cnn_denoiser", file=sys.stderr)
    else:
        print(f"warning: checkpoint not found at {checkpoint}; skipping cnn_denoiser", file=sys.stderr)

    dataset = Path(args.dataset)
    if not (dataset / "manifest.csv").exists():
        print(f"error: manifest not found at {dataset / 'manifest.csv'}", file=sys.stderr)
        print("  run: python scripts/generate_denoising_dataset.py", file=sys.stderr)
        return 1

    report = evaluate_denoiser(
        dataset,
        config,
        inferencer,
        split=args.split,
        verbose=not args.quiet,
    )

    # Print overall summary table
    print(f"\n{'='*60}")
    print(f"Strategy comparison — {args.split} split  ({report.n_test} samples)")
    print(f"{'='*60}")
    print(f"  {'strategy':<16} {'MSE':>8}  {'PSNR':>8}  {'SSIM':>6}")
    print(f"  {'-'*16} {'-'*8}  {'-'*8}  {'-'*6}")
    for name, sr in report.strategies.items():
        psnr_str = f"{sr.overall_psnr:8.2f}" if sr.overall_psnr < 1e14 else "     inf"
        print(f"  {name:<16} {sr.overall_mse:8.5f}  {psnr_str}  {sr.overall_ssim:6.4f}")

    # Per-level tables
    print(f"\n{'-'*60}")
    print("Per-level breakdown (Δ vs no_filter baseline)")
    print(f"{'-'*60}")
    for strategy in report.strategies:
        _print_level_table(report, strategy)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"\nJSON report written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
