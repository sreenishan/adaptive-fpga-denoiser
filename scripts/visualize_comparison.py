#!/usr/bin/env python3
"""
scripts/visualize_comparison.py

Produces two publication-quality figure formats:

  Format 1 — comparison grid:
    Rows = test images/noise combinations.
    Columns = Median | Gaussian | Wiener | "Ours" (adaptive) | Ground Truth.
    Each cell shows the image with a red inset crop and PSNR (dB) below it.
    Best PSNR per row is bolded.

  Format 2 — triplet:
    Original | Noisy image | Denoised image  (single test case).

Outputs are saved to results/comparison/.

Usage:
    python scripts/visualize_comparison.py
    python scripts/visualize_comparison.py --noise gaussian --output results/comparison
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from denoising.config import load_hardware_config, load_inference_config
from denoising.filters import apply_filter
from denoising.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
)
from denoising.metrics import calculate_psnr as compute_psnr, calculate_ssim as compute_ssim

# ── Config ────────────────────────────────────────────────────────────────────
hw_cfg  = load_hardware_config()
inf_cfg = load_inference_config()

NOISE_PARAMS: dict[str, dict] = {
    "salt_pepper": {"amount": 0.05},
    "gaussian":    {"sigma": 0.15},
    "speckle":     {"variance": 0.04},
}

def _add_noise(noise_type: str, image: np.ndarray, seed: int, **params) -> np.ndarray:
    if noise_type == "salt_pepper":
        return add_salt_pepper_noise(image, seed=seed, **params)
    if noise_type == "gaussian":
        return add_gaussian_noise(image, seed=seed, **params)
    if noise_type == "speckle":
        return add_speckle_noise(image, seed=seed, **params)
    raise ValueError(f"unknown noise type: {noise_type}")

NOISE_LABEL = {
    "salt_pepper": "Salt & Pepper",
    "gaussian":    "Gaussian",
    "speckle":     "Speckle",
}

# Filters applied for each noise type in Format 1
# Order: (a) class-specific  (b) neighbour method  (c) wiener  (d) Ours
FILTER_PLAN: dict[str, list[tuple[str, str]]] = {
    "salt_pepper": [
        ("Median",          "median"),
        ("Gaussian",        "gaussian"),
        ("Wiener",          "wiener"),
        ("Adaptive Median", "adaptive_median"),
    ],
    "gaussian": [
        ("Median",          "median"),
        ("Gaussian",        "gaussian"),
        ("Wiener",          "wiener"),
        ("Adaptive Median", "adaptive_median"),
    ],
    "speckle": [
        ("Median",          "median"),
        ("Gaussian",        "gaussian"),
        ("Wiener",          "wiener"),
        ("Adaptive Median", "adaptive_median"),
    ],
}

FILTER_KWARGS: dict[str, dict] = {
    "median":          {"kernel_size": 3},
    "gaussian":        {"kernel_size": 3},
    "wiener":          {"kernel_size": 3, "noise_variance": None},
    "adaptive_median": {"kernel_size": 3, "max_size": 7},
}


# ── Test case specification ───────────────────────────────────────────────────
class TestCase(NamedTuple):
    label: str          # e.g. "CT (salt-pepper)"
    clean: np.ndarray   # uint8 224×224
    noise_type: str
    # Crop ROI: (row_start, col_start, row_end, col_end)
    crop: tuple[int, int, int, int]


def _pick_images(n: int = 4) -> list[tuple[str, np.ndarray]]:
    """Return up to n clean 224×224 images from the raw dataset."""
    sources = [
        ("CT",          PROJECT_ROOT / "data" / "raw" / "ct"),
        ("MRI",         PROJECT_ROOT / "data" / "raw" / "mri"),
        ("Ultrasound",  PROJECT_ROOT / "data" / "raw" / "ultrasound"),
    ]
    images: list[tuple[str, np.ndarray]] = []
    rng = np.random.default_rng(42)
    for name, folder in sources:
        if not folder.exists():
            continue
        files = sorted(folder.glob("*.png"))
        if not files:
            continue
        chosen = rng.choice(files, size=min(2, len(files)), replace=False)
        for path in chosen:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (224, 224):
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            images.append((name, img.copy()))
            if len(images) >= n:
                return images
    return images


def _make_test_cases(n_images: int = 4) -> list[TestCase]:
    images = _pick_images(n_images)
    noise_types = ["salt_pepper", "gaussian", "speckle"]
    cases: list[TestCase] = []
    rng = np.random.default_rng(7)
    for i, (src_name, clean) in enumerate(images):
        ntype = noise_types[i % len(noise_types)]
        # Pick a visually interesting crop (avoid flat corners)
        # Default to top-left quadrant with some variety
        offsets = [(30, 30), (20, 100), (80, 60), (50, 140)]
        r0, c0 = offsets[i % len(offsets)]
        crop = (r0, c0, r0 + 70, c0 + 70)
        label = f"{src_name} ({NOISE_LABEL[ntype]})"
        cases.append(TestCase(label=label, clean=clean,
                               noise_type=ntype, crop=crop))
    return cases


# ── Adaptive "Ours" selector ──────────────────────────────────────────────────
def ours(noisy: np.ndarray, noise_type: str) -> np.ndarray:
    """Apply the adaptive pipeline filter for the given noise class."""
    from denoising.filters.selector import FILTER_FOR_CLASS, SEVERITY_POLICY
    from denoising.severity import estimate_severity

    filter_name = FILTER_FOR_CLASS[noise_type]
    severity = estimate_severity(noisy, noise_type, inf_cfg.severity)
    if severity is not None:
        step = SEVERITY_POLICY.get((noise_type, severity))
        if step is not None:
            filter_name = step.filter_name
            kwargs = dict(FILTER_KWARGS.get(filter_name, {}))
            result = noisy.copy()
            for _ in range(step.passes):
                result = apply_filter(result, filter_name, **kwargs)
            return result

    kwargs = dict(FILTER_KWARGS.get(filter_name, {}))
    return apply_filter(noisy, filter_name, **kwargs)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _draw_inset(ax: "plt.Axes", img: np.ndarray,
                crop: tuple[int, int, int, int],
                border_color: str = "red") -> None:
    """Show img on ax, draw a red crop box, and embed the cropped inset."""
    r0, c0, r1, c1 = crop
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="equal")
    # Red box on main image
    rect = patches.Rectangle(
        (c0, r0), c1 - c0, r1 - r0,
        linewidth=1.5, edgecolor=border_color, facecolor="none"
    )
    ax.add_patch(rect)
    # Inset axes: bottom-right corner, ~28% of the image
    h, w = img.shape
    ins_w = 0.32
    ins_ax = ax.inset_axes([1 - ins_w - 0.01, 0.01, ins_w, ins_w])
    crop_img = img[r0:r1, c0:c1]
    ins_ax.imshow(crop_img, cmap="gray", vmin=0, vmax=255, aspect="equal")
    ins_ax.set_xticks([])
    ins_ax.set_yticks([])
    for spine in ins_ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _label_str(name: str, psnr: float | None, bold: bool) -> str:
    if psnr is None:
        return name
    val = f"{psnr:.2f} dB"
    if bold:
        return f"{name} ($\\mathbf{{{val}}}$)"
    return f"{name} ({val})"


# ── Format 1: comparison grid ─────────────────────────────────────────────────
def make_comparison_grid(cases: list[TestCase], out_dir: Path,
                          seed: int = 0) -> Path:
    """5-column comparison: (a)..(d) individual filters, (d) Ours, (e) GT."""
    n_rows = len(cases)
    n_cols = 5   # 4 methods + Ground Truth
    col_labels_base = ["(a)", "(b)", "(c)", "(d) Ours", "(e) Ground Truth"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.2 * n_rows),
        squeeze=False
    )
    fig.patch.set_facecolor("white")

    for row_idx, case in enumerate(cases):
        noisy = _add_noise(
            case.noise_type, case.clean,
            seed=seed + row_idx,
            **NOISE_PARAMS[case.noise_type],
        )
        plan = FILTER_PLAN[case.noise_type]   # list of (label, filter_name)

        # Build outputs for columns 0..2 (individual filters)
        filter_outputs: list[tuple[str, np.ndarray]] = []
        for _, fname in plan[:3]:
            kwargs = dict(FILTER_KWARGS.get(fname, {}))
            out = apply_filter(noisy, fname, **kwargs)
            filter_outputs.append((fname, out))

        # Column 3: "Ours"
        ours_out = ours(noisy, case.noise_type)
        filter_outputs.append(("ours", ours_out))

        # Compute PSNRs (vs clean)
        psnrs: list[float | None] = []
        for _, out in filter_outputs:
            psnrs.append(compute_psnr(case.clean, out))

        best_psnr = max(p for p in psnrs if p is not None)

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < 4:
                fname, out_img = filter_outputs[col_idx]
                method_label = plan[col_idx][0] if col_idx < 3 else "Ours"
                psnr = psnrs[col_idx]
                bold = (psnr is not None and abs(psnr - best_psnr) < 0.005)
                letter = ["(a)", "(b)", "(c)", "(d)"][col_idx]
                _draw_inset(ax, out_img, case.crop)
                ax.set_xlabel(
                    f"{letter} {method_label}"
                    + (f" ($\\mathbf{{{psnr:.2f}}}$ dB)" if bold and psnr is not None
                       else (f" ({psnr:.2f} dB)" if psnr is not None else "")),
                    fontsize=8, labelpad=3
                )
            else:
                # Ground truth
                _draw_inset(ax, case.clean, case.crop)
                ax.set_xlabel("(e) Ground Truth", fontsize=8, labelpad=3)

            if col_idx == 0:
                ax.set_ylabel(case.label, fontsize=8, rotation=90, labelpad=4)

    plt.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.2)
    out_path = out_dir / "comparison_grid.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ── Format 2: triplet ─────────────────────────────────────────────────────────
def make_triplet(case: TestCase, out_dir: Path, seed: int = 0) -> Path:
    """Original | Noisy image | Denoised image."""
    noisy = _add_noise(
        case.noise_type, case.clean,
        seed=seed, **NOISE_PARAMS[case.noise_type]
    )
    denoised = ours(noisy, case.noise_type)

    psnr_noisy   = compute_psnr(case.clean, noisy)
    psnr_denoised = compute_psnr(case.clean, denoised)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    fig.patch.set_facecolor("white")

    titles = ["Original", "Noisy image", "Denoised image"]
    imgs   = [case.clean, noisy, denoised]
    psnrs  = [None, psnr_noisy, psnr_denoised]

    for ax, img, title, psnr in zip(axes, imgs, titles, psnrs):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        label = title
        if psnr is not None:
            label += f"\n({psnr:.2f} dB)"
        ax.set_title(label, fontsize=11, pad=4)

    plt.tight_layout(pad=0.5)
    out_path = out_dir / "triplet.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate comparison figures")
    ap.add_argument("--output", default="results/comparison",
                    help="Output directory (default: results/comparison)")
    ap.add_argument("--n-images", type=int, default=4,
                    help="Number of test images (default: 4)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for noise generation")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Picking {args.n_images} test images …")
    cases = _make_test_cases(args.n_images)
    if not cases:
        print("ERROR: no images found in data/raw/. Run scripts/fetch_medical_sources.py first.")
        sys.exit(1)

    for i, c in enumerate(cases):
        print(f"  [{i+1}] {c.label}  (clean shape={c.clean.shape})")

    print("\nGenerating Format 1: comparison grid …")
    make_comparison_grid(cases, out_dir, seed=args.seed)

    print("\nGenerating Format 2: triplet (first test case) …")
    make_triplet(cases[0], out_dir, seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
