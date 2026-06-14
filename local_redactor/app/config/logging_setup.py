"""Status-only logging setup.

Hard Invariant: logs hold settings, file names, counts, and status only — never
document text or detected PII. The logger is configured to INFO by default;
DEBUG is opt-in and explicitly warned about (see secure_temp.warn_if_debug_logging).
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from app.config.paths import logs_dir

_CONFIGURED = False


def configure_logging(debug: bool = False) -> logging.Logger:
    """Configure the `localredactor` logger once; rotate a status log on disk."""
    global _CONFIGURED
    logger = logging.getLogger("localredactor")
    if _CONFIGURED:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        return logger

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        fileh = RotatingFileHandler(
            logs_dir() / "localredactor.log", maxBytes=512_000, backupCount=3, encoding="utf-8"
        )
        fileh.setFormatter(fmt)
        logger.addHandler(fileh)
    except OSError:  # pragma: no cover - read-only/locked profile dir
        pass

    _CONFIGURED = True
    return logger
