#!/usr/bin/env python3
"""Generate the four-class noise dataset and its manifest.

Usage:
    python scripts/generate_dataset.py                 # from data/raw/
    python scripts/generate_dataset.py --synthetic 20  # deterministic patterns
    python scripts/generate_dataset.py --dry-run       # print the plan only
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a checkout without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoising.dataset.cli import main  # noqa: E402  (path set up above)

if __name__ == "__main__":
    raise SystemExit(main())
