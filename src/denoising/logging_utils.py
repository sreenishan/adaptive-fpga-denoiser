"""Project-wide logging setup.

Scripts call :func:`configure_logging` once at start-up; library modules only
ever call :func:`get_logger` and never touch handlers, so importing the package
cannot hijack an application's logging configuration.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

__all__ = ["LOG_FORMAT", "configure_logging", "get_logger"]

LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT: Final[str] = "%H:%M:%S"
_ROOT_LOGGER_NAME: Final[str] = "denoising"


def configure_logging(level: int | str = logging.INFO, *, force: bool = False) -> logging.Logger:
    """Attach a single stderr handler to the ``denoising`` logger.

    Args:
        level: Logging level, as an int or a name such as ``"DEBUG"``.
        force: Replace any handler this function added previously. Without it,
            calling twice is a no-op rather than a duplicated log line.

    Returns:
        The configured ``denoising`` logger.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if logger.handlers and not force:
        logger.setLevel(level)
        return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``denoising`` namespace.

    ``get_logger(__name__)`` from inside the package returns that module's
    logger unchanged; any other name is nested under ``denoising``.
    """
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
