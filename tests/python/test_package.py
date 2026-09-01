"""Package import surface and logging setup (spec section 6, phase 1)."""

from __future__ import annotations

import logging
import sys

import denoising
from denoising.logging_utils import configure_logging, get_logger


def test_package_imports_without_a_ml_framework() -> None:
    """Importing the package must not pull in torch: it is an optional extra.

    We verify that ``denoising/__init__.py`` itself does not import from
    ``denoising.model`` (which pulls in torch).  The check is done on the
    source rather than ``sys.modules`` because the full test suite runs in one
    process: if ``test_model.py`` ran first, torch is already loaded regardless
    of whether this import pulled it in.
    """
    import ast
    from pathlib import Path

    assert denoising.__version__ == "0.1.0"

    init_src = (Path(denoising.__file__)).read_text(encoding="utf-8")
    tree = ast.parse(init_src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            for alias in getattr(node, "names", []):
                name = getattr(alias, "name", "") or ""
            assert "torch" not in module, f"denoising/__init__.py imports torch: {ast.dump(node)}"
            assert "model" not in module, f"denoising/__init__.py imports from .model: {ast.dump(node)}"


def test_class_order_is_the_label_order() -> None:
    assert denoising.CLASSES == ("clean", "salt_pepper", "gaussian", "speckle")
    assert denoising.CLASSES.index("clean") == 0


def test_configure_logging_is_idempotent() -> None:
    first = configure_logging(logging.DEBUG)
    handlers = len(first.handlers)
    second = configure_logging(logging.INFO)
    assert second is first
    assert len(second.handlers) == handlers


def test_get_logger_namespaces_under_the_package() -> None:
    assert get_logger("noise.salt_pepper").name == "denoising.noise.salt_pepper"
    assert get_logger("denoising.filters").name == "denoising.filters"
