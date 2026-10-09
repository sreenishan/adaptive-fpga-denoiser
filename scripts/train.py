"""CLI entry point: train the noise classifier (spec phase 7).

Usage::

    python scripts/train.py --dataset data/processed
    python scripts/train.py --dataset data/processed --epochs 10 --device cpu
    python scripts/train.py --dataset data/processed --resume

The script loads ``configs/training.yaml``, overrides any flags supplied on the
command line, runs the training loop and prints per-epoch progress.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without `pip install -e .`
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the adaptive noise classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Directory containing manifest.csv (output of denoising-generate-dataset).",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    p.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    p.add_argument("--lr", type=float, default=None, help="Override training.learning_rate.")
    p.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override training.device.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override training.checkpoint_dir.",
    )
    p.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping regardless of config.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-epoch output.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse()

    from denoising.config import load_training_config
    from denoising.model.train import train

    config = load_training_config()

    # Apply CLI overrides. TrainingConfig is a frozen dataclass so we use
    # object.__setattr__ to reach past the freeze.
    if args.epochs is not None:
        object.__setattr__(config, "epochs", args.epochs)
    if args.batch_size is not None:
        object.__setattr__(config, "batch_size", args.batch_size)
    if args.lr is not None:
        object.__setattr__(config, "learning_rate", args.lr)
    if args.device is not None:
        object.__setattr__(config, "device", args.device)
    if args.checkpoint_dir is not None:
        object.__setattr__(config, "checkpoint_dir", str(args.checkpoint_dir))
    if args.no_early_stopping:
        object.__setattr__(config.early_stopping, "enabled", False)

    print("=" * 60)
    print("  Adaptive FPGA Noise Classifier — Training")
    print("=" * 60)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Epochs    : {config.epochs}")
    print(f"  Batch     : {config.batch_size}")
    print(f"  LR        : {config.learning_rate}")
    print(f"  Optimizer : {config.optimizer}")
    print(f"  Scheduler : {config.scheduler}")
    print(f"  Device    : {config.device}")
    print(f"  Early stop: {config.early_stopping.enabled} (patience {config.early_stopping.patience})")
    print("=" * 60)

    result = train(config, args.dataset, verbose=not args.quiet)

    print("\n" + "=" * 60)
    print(f"  Best val accuracy : {result.best_val_acc:.4f} ({result.best_val_acc*100:.2f}%)")
    print(f"  Best epoch        : {result.best_epoch}")
    print(f"  Stopped early     : {result.stopped_early}")
    print(f"  Checkpoint        : {result.checkpoint_path}")
    print(f"  Metadata          : {result.metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
