"""Compute safe output paths — always `_redacted`, never overwrite the input.

Hard Invariant: outputs go to a separate location and never overwrite.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def redacted_output_path(
    source: Path, out_dir: Optional[Path] = None, suffix: str = "_redacted"
) -> Path:
    """Return `<name>_redacted.pdf` in `out_dir` (or alongside the source).

    If a file with the target name already exists, a numeric counter is added
    so we never overwrite an existing output either.
    """
    source = Path(source)
    out_dir = Path(out_dir) if out_dir else source.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Redacted output is always a PDF, regardless of the input being an image.
    stem = source.stem + suffix
    candidate = out_dir / f"{stem}.pdf"
    counter = 1
    while candidate.exists():
        candidate = out_dir / f"{stem}_{counter}.pdf"
        counter += 1
    return candidate
