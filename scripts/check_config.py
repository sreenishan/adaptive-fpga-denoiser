#!/usr/bin/env python3
"""Validate configs/*.yaml and print what was parsed.

Usage:
    python scripts/check_config.py
    python scripts/check_config.py --config-dir path/to/configs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a checkout without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoising.cli import main  # noqa: E402  (path set up above)

if __name__ == "__main__":
    raise SystemExit(main())
