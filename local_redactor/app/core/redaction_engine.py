"""Compute boxes, burn them in, and build the redacted output.

Mode A (default): flatten every page to a high-DPI image, paint boxes onto the
pixels, assemble an image-only PDF with NO text layer (img2pdf). Copying yields
nothing by construction.

Mode B (optional, born-digital only): PyMuPDF stream redaction that truly
removes content rather than covering it.

Whichever mode runs, all metadata is stripped on export.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List

import fitz  # PyMuPDF
import img2pdf
from PIL import Image, ImageDraw

from app.core.pdf_renderer import scale_for_dpi
from app.detectors.base import Box

# PDF metadata keys we always clear on export.
_METADATA_KEYS = [
    "title", "author", "subject", "keywords", "creator", "producer",
    "format", "encryption", "creationDate", "modDate", "trapped",
]


@dataclass
class PageRedaction:
    """Boxes to paint on one page, expressed in the named coordinate space."""

    boxes: List[Box] = field(default_factory=list)
    #: "pixels" (image device pixels at render DPI) or "points" (PDF user space).
    space: str = "pixels"


# --------------------------------------------------------------------------- #
# Mode A — flatten to image.
# --------------------------------------------------------------------------- #
def burn_boxes_on_image(img: Image.Image, boxes: List[Box]) -> Image.Image:
    """Paint solid black rectangles onto a copy of `img` (boxes in pixels)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for b in boxes:
        draw.rectangle([b.x0, b.y0, b.x1, b.y1], fill="black")
    return out


def _encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_image_only_pdf(images: List[Image.Image]) -> bytes:
    """Assemble flattened images into an image-only PDF with no text layer."""
    if not images:
        raise ValueError("Cannot build a PDF from zero pages.")
    encoded = [_encode_png(im) for im in images]
    # img2pdf produces a PDF whose pages are purely the images: no text, no
    # metadata beyond what we strip below.
    return img2pdf.convert(encoded)


def to_pixel_boxes(boxes: List[Box], space: str, dpi: int) -> List[Box]:
    """Convert point-space boxes to pixel space if needed (Mode A draws pixels)."""
    if space == "pixels":
        return boxes
    s = scale_for_dpi(dpi)
    return [Box(b.x0 * s, b.y0 * s, b.x1 * s, b.y1 * s) for b in boxes]


# --------------------------------------------------------------------------- #
# Mode B — structure-preserving stream redaction (born-digital only).
# --------------------------------------------------------------------------- #
def apply_mode_b(doc: "fitz.Document", page_boxes: Dict[int, List[Box]]) -> bytes:
    """Redact a born-digital doc in place on a copy; boxes in PDF points.

    Caller must guarantee every page is born-digital (page_classifier).
    """
    for page_index, boxes in page_boxes.items():
        page = doc[page_index]
        for b in boxes:
            page.add_redact_annot(fitz.Rect(b.x0, b.y0, b.x1, b.y1), fill=(0, 0, 0))
        # Remove the underlying characters, not just cover them.
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        )
    _strip_metadata(doc)
    return doc.tobytes(garbage=4, deflate=True, clean=True)


def _strip_metadata(doc: "fitz.Document") -> None:
    """Clear document metadata, XMP, embedded files and JavaScript."""
    doc.set_metadata({k: "" for k in _METADATA_KEYS})
    try:
        doc.del_xml_metadata()
    except Exception:  # pragma: no cover - not all docs carry XMP
        pass
    # Remove embedded files.
    try:
        for name in list(doc.embfile_names()):
            doc.embfile_del(name)
    except Exception:  # pragma: no cover
        pass
    # Drop document-level JavaScript if present.
    try:
        for xref in range(1, doc.xref_length()):
            if doc.xref_get_key(xref, "S")[1] == "/JavaScript":
                doc.update_object(xref, "<<>>")
    except Exception:  # pragma: no cover
        pass


def strip_pdf_metadata_bytes(pdf_bytes: bytes) -> bytes:
    """Reopen a PDF bytes blob and strip its metadata (used after img2pdf)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        _strip_metadata(doc)
        return doc.tobytes(garbage=4, deflate=True, clean=True)
    finally:
        doc.close()
