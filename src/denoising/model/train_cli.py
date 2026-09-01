"""Console-script entry point for ``denoising-train``."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> None:
    # Re-use the standalone script logic by importing and calling it.
    from scripts.train import main as _main  # type: ignore[import]

    _main()
