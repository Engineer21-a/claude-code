"""Secure-enough temporary file handling.

Phase 4 guidance: prefer in-memory page images, touch the OS temp dir only when
necessary, wipe the job workdir on completion, and warn loudly if debug logging
is enabled (which could spill document text to disk).

The main pipeline already works entirely in memory (PIL images + bytes), so this
is used only when a backend genuinely needs a path on disk. The context manager
best-effort overwrites file contents before unlinking so deleted scratch data is
not trivially recoverable.
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("localredactor")


@contextmanager
def secure_workdir(prefix: str = "localredactor_") -> Iterator[Path]:
    """Create a temp working directory and wipe it (overwrite + remove) on exit."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        _wipe_dir(path)


def _wipe_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if child.is_file() or child.is_symlink():
                _overwrite_file(child)
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        except OSError:  # pragma: no cover - best effort wipe
            pass
    try:
        path.rmdir()
    except OSError:  # pragma: no cover
        pass


def _overwrite_file(path: Path) -> None:
    """Overwrite a file's bytes with zeros before deletion (best effort)."""
    try:
        size = path.stat().st_size
        if size:
            with path.open("r+b", buffering=0) as f:
                f.write(b"\x00" * size)
                f.flush()
                os.fsync(f.fileno())
    except OSError:  # pragma: no cover - device-dependent
        pass


def warn_if_debug_logging(debug_logging: bool) -> list[str]:
    """Return a warning list if debug logging is on (it may capture text)."""
    warnings: list[str] = []
    if debug_logging or logger.isEnabledFor(logging.DEBUG):
        msg = (
            "Debug logging is enabled — document text may be written to disk. "
            "Disable it for privacy-critical work."
        )
        logger.warning(msg)
        warnings.append(msg)
    return warnings
