"""The configuration CLI (spec section 6, phase 1 acceptance criteria)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from denoising import config as cfg
from denoising.cli import main

_CONFIG_FILES = ("dataset.yaml", "training.yaml", "inference.yaml", "hardware.yaml")


def test_cli_reports_success_on_the_shipped_configs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "dataset.yaml" in out
    assert "hardware.yaml" in out
    # An unconfigured board is shown as TBD, never as a plausible-looking number.
    assert "no board configured (TBD)" in out


def test_cli_exits_non_zero_on_a_bad_config(tmp_path: Path) -> None:
    for name in _CONFIG_FILES:
        shutil.copy(cfg.CONFIG_DIR / name, tmp_path / name)
    broken = tmp_path / "inference.yaml"
    text = broken.read_text(encoding="utf-8").replace("threshold: 0.60", "threshold: 4.0")
    broken.write_text(text, encoding="utf-8")
    assert main(["--config-dir", str(tmp_path)]) == 1


def test_cli_exits_non_zero_when_the_directory_is_empty(tmp_path: Path) -> None:
    assert main(["--config-dir", str(tmp_path)]) == 1
