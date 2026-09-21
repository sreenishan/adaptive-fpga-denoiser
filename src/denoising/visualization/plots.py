"""Visualization helpers for the DnCNN denoising experiment.

All functions return ``matplotlib`` figure objects — they never call
``plt.show()`` or write to disk, so callers control output.

Functions
---------
plot_training_curves          Loss and PSNR vs epoch from a metadata JSON.
plot_metrics_vs_level         PSNR/SSIM/MSE vs noise level for each strategy.
plot_image_grid               Original | Noisy | Denoised comparison grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "plot_training_curves",
    "plot_metrics_vs_level",
    "plot_image_grid",
]


def _require_mpl() -> Any:
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualization") from exc


# ─── Training curves ──────────────────────────────────────────────────────────


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    val_psnrs: Sequence[float] | None = None,
    *,
    title: str = "DnCNN Training Curves",
) -> Any:
    """Plot training/validation loss and optionally validation PSNR vs epoch.

    Args:
        train_losses: Per-epoch training MSE loss.
        val_losses:   Per-epoch validation MSE loss.
        val_psnrs:    Per-epoch validation PSNR in dB (optional).
        title:        Figure title.

    Returns:
        ``matplotlib.figure.Figure``
    """
    plt = _require_mpl()
    n_rows = 2 if val_psnrs is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    epochs = list(range(1, len(train_losses) + 1))
    axes[0].plot(epochs, train_losses, label="train loss", color="steelblue")
    axes[0].plot(epochs, val_losses, label="val loss", color="coral", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if val_psnrs is not None:
        axes[1].plot(epochs, val_psnrs, color="seagreen")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("PSNR (dB)")
        axes[1].set_title("Validation PSNR")
        axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ─── Metrics vs noise level ───────────────────────────────────────────────────


def plot_metrics_vs_level(
    report: Any,
    noise_type: str,
    *,
    metrics: Sequence[str] = ("psnr", "ssim", "mse"),
    title: str | None = None,
) -> Any:
    """Line chart of PSNR/SSIM/MSE vs noise level for each strategy.

    Args:
        report:     A :class:`~denoising.evaluation.denoising_eval.DenoisingEvalReport`.
        noise_type: One of ``salt_pepper``, ``gaussian``, ``speckle``.
        metrics:    Which metrics to plot (subset of psnr/ssim/mse).
        title:      Figure title; auto-generated if None.

    Returns:
        ``matplotlib.figure.Figure``
    """
    plt = _require_mpl()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    strategy_colors = {
        "no_filter": "gray",
        "fixed_median": "steelblue",
        "gt_adaptive": "seagreen",
        "cnn_denoiser": "firebrick",
    }

    for strategy, level_list in report.per_level.items():
        rows = [lv for lv in level_list if lv.noise_type == noise_type]
        if not rows:
            continue
        rows = sorted(rows, key=lambda x: x.noise_level)
        levels = [r.noise_level for r in rows]
        color = strategy_colors.get(strategy, "purple")

        for ax, metric in zip(axes, metrics):
            vals = [getattr(r, metric) for r in rows]
            # Skip infinite PSNR values in plot
            if metric == "psnr":
                vals = [v if v < 1e14 else float("nan") for v in vals]
            ax.plot(levels, vals, marker="o", label=strategy, color=color)

    for ax, metric in zip(axes, metrics):
        ax.set_xlabel("Noise level (%)")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xticks([5, 10, 15, 20])

    t = title or f"Metrics vs Noise Level — {noise_type.replace('_', ' ').title()}"
    fig.suptitle(t, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ─── Image comparison grid ────────────────────────────────────────────────────


def plot_image_grid(
    clean: np.ndarray,
    noisy_images: Sequence[tuple[str, np.ndarray]],
    denoised_images: Sequence[tuple[str, np.ndarray]],
    *,
    title: str = "Denoising Comparison",
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Show Original | Noisy | Denoised for each condition.

    Args:
        clean:            The clean reference image (H, W) uint8.
        noisy_images:     List of (label, image) for the noisy column.
        denoised_images:  List of (label, image) for the denoised column; must
                          match *noisy_images* in length.
        title:            Figure title.
        figsize:          Override figure size.

    Returns:
        ``matplotlib.figure.Figure``
    """
    plt = _require_mpl()
    n_rows = max(len(noisy_images), 1)
    n_cols = 3  # clean | noisy | denoised
    w, h = figsize or (4 * n_cols, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w, h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    cmap = "gray"

    for row_idx in range(n_rows):
        # Clean column
        axes[row_idx, 0].imshow(clean, cmap=cmap, vmin=0, vmax=255)
        axes[row_idx, 0].set_title("Clean" if row_idx == 0 else "")
        axes[row_idx, 0].axis("off")

        # Noisy column
        if row_idx < len(noisy_images):
            noisy_label, noisy_img = noisy_images[row_idx]
            axes[row_idx, 1].imshow(noisy_img, cmap=cmap, vmin=0, vmax=255)
            axes[row_idx, 1].set_title(f"Noisy: {noisy_label}")
            axes[row_idx, 1].axis("off")

        # Denoised column
        if row_idx < len(denoised_images):
            den_label, den_img = denoised_images[row_idx]
            axes[row_idx, 2].imshow(den_img, cmap=cmap, vmin=0, vmax=255)
            axes[row_idx, 2].set_title(f"Denoised: {den_label}")
            axes[row_idx, 2].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
