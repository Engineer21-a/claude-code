"""Read-only document loading, hashing, and type detection.

Hard Invariant: never write/move/delete an input file. Everything here opens
sources read-only and only ever reads bytes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DocType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}

# Magic-number sniffing so we do not trust extensions alone.
_PDF_MAGIC = b"%PDF-"
_IMAGE_MAGICS = (
    b"\x89PNG\r\n\x1a\n",     # PNG
    b"\xff\xd8\xff",          # JPEG
    b"BM",                    # BMP
    b"II*\x00",               # TIFF little-endian
    b"MM\x00*",               # TIFF big-endian
    b"GIF87a",
    b"GIF89a",
)


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    doc_type: DocType
    sha256: str
    size_bytes: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    # Read-only, streamed so large files don't blow up memory.
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_type(path: Path) -> DocType:
    try:
        with path.open("rb") as f:
            head = f.read(16)
    except OSError:
        return DocType.UNKNOWN

    if head.startswith(_PDF_MAGIC):
        return DocType.PDF
    if any(head.startswith(m) for m in _IMAGE_MAGICS):
        return DocType.IMAGE
    # Fall back to the extension only when magic bytes are inconclusive.
    if path.suffix.lower() == ".pdf":
        return DocType.PDF
    if path.suffix.lower() in _IMAGE_EXTS:
        return DocType.IMAGE
    return DocType.UNKNOWN


def load(path: str | Path) -> LoadedDocument:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return LoadedDocument(
        path=path,
        doc_type=detect_type(path),
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )
