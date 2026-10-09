"""Command line entry point for configuration validation.

``python scripts/check_config.py`` (or ``denoising-check-config`` once the
package is installed) loads every file in ``configs/`` and prints what was
parsed. It exits non-zero on the first invalid value, which makes it usable as
a pre-flight check in CI before any long-running phase.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .config import (
    CONFIG_DIR,
    PROJECT_ROOT,
    ConfigError,
    load_dataset_config,
    load_hardware_config,
    load_inference_config,
    load_training_config,
)
from .logging_utils import configure_logging, get_logger

__all__ = ["main"]

_LOG = get_logger(__name__)


def _relative(path: Path) -> str:
    """Show a path relative to the repository root when it lives inside it."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _report(config_dir: Path) -> None:
    """Load all four configuration files and print a one-screen summary."""
    dataset = load_dataset_config(config_dir / "dataset.yaml")
    training = load_training_config(config_dir / "training.yaml")
    inference = load_inference_config(config_dir / "inference.yaml")
    hardware = load_hardware_config(config_dir / "hardware.yaml")

    print(f"config directory : {_relative(config_dir)}")
    print(f"project root     : {PROJECT_ROOT}")
    print()
    print("dataset.yaml")
    print(f"  image          : {dataset.image.width}x{dataset.image.height}"
          f" grayscale={dataset.image.grayscale}")
    print(f"  split          : train={dataset.split.train_ratio}"
          f" val={dataset.split.validation_ratio}"
          f" test={dataset.split.test_ratio} seed={dataset.split.seed}")
    print(f"  salt & pepper  : amounts={list(dataset.noise.salt_pepper.amounts)}"
          f" salt_vs_pepper={dataset.noise.salt_pepper.salt_vs_pepper}")
    print(f"  gaussian       : mean={dataset.noise.gaussian.mean}"
          f" sigmas={list(dataset.noise.gaussian.sigmas)}")
    print(f"  speckle        : variances={list(dataset.noise.speckle.variances)}")
    print(f"  manifest       : {_relative(dataset.paths.manifest)}")
    print()
    print("training.yaml")
    print(f"  model          : {training.model.num_classes} classes,"
          f" {training.model.input_channels} channel(s),"
          f" base={training.model.base_channels}, dropout={training.model.dropout}")
    print(f"  optimisation   : {training.optimizer} lr={training.learning_rate}"
          f" batch={training.batch_size} epochs={training.epochs}"
          f" scheduler={training.scheduler}")
    print(f"  early stopping : enabled={training.early_stopping.enabled}"
          f" patience={training.early_stopping.patience}")
    print(f"  checkpoints    : {_relative(training.checkpoint_dir)}")
    print()
    print("inference.yaml")
    print(f"  model          : {_relative(inference.model_path)}"
          f" (present={inference.model_path.is_file()})")
    print(f"  confidence     : threshold={inference.confidence.threshold}"
          f" fallback={inference.confidence.fallback}")
    print(f"  filters        : boundary={inference.filters.boundary_mode}"
          f" median={inference.filters.median.kernel_size}"
          f" gaussian={inference.filters.gaussian.kernel_size}"
          f"/sigma={inference.filters.gaussian.sigma}"
          f" wiener={inference.filters.wiener.kernel_size}"
          f"/noise_variance={inference.filters.wiener.noise_variance}")
    print()
    print("hardware.yaml")
    print(f"  stream         : {hardware.stream.pixel_width}-bit"
          f" {hardware.stream.image_width}x{hardware.stream.image_height}"
          f" boundary={hardware.stream.boundary_policy}"
          f" backpressure={hardware.stream.backpressure}")
    print(f"  simulation     : {hardware.simulation.simulator}"
          f" tolerance={dict(hardware.simulation.max_abs_error)}")
    if hardware.synthesis.configured:
        print(f"  synthesis      : {hardware.synthesis.vendor}"
              f" {hardware.synthesis.device}"
              f" clock={hardware.synthesis.clock_mhz} MHz"
              f" tool={hardware.synthesis.tool_version}")
    else:
        print("  synthesis      : no board configured (TBD)")


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the configuration files. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="denoising-check-config",
        description="Load and validate every file in configs/.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help=f"directory holding the YAML files (default: {_relative(CONFIG_DIR)})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    configure_logging(getattr(logging, args.log_level))
    try:
        _report(args.config_dir)
    except ConfigError as exc:
        _LOG.error("invalid configuration: %s", exc)
        return 1
    _LOG.info("all four configuration files loaded and validated")
    return 0
