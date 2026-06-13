"""Shared pytest fixtures: tiny born-digital and scanned PDF samples.

Generated at runtime with PyMuPDF/Pillow so the repo carries no binary blobs.
The fixtures intentionally contain a small set of German PII-shaped strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the package importable when running `pytest` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fitz = pytest.importorskip("fitz", reason="PyMuPDF required for sample fixtures")


SECRET_NAME = "Erika Mustermann"
SECRET_IBAN = "DE89 3704 0044 0532 0130 00"
SECRET_EMAIL = "erika.mustermann@example.de"
BORN_DIGITAL_LINES = [
    "Rechnung Nr. 2026-00042",
    f"Kundin: {SECRET_NAME}",
    f"IBAN: {SECRET_IBAN}",
    f"E-Mail: {SECRET_EMAIL}",
    "Betrag: 1.234,56 EUR",
]


@pytest.fixture
def born_digital_pdf(tmp_path: Path) -> Path:
    """A real text-layer PDF containing known secrets."""
    path = tmp_path / "born_digital.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in BORN_DIGITAL_LINES:
        page.insert_text((72, y), line, fontsize=12)
        y += 24
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """An image-only PDF (no text layer): the born-digital page rasterised."""
    src = tmp_path / "for_scan.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in BORN_DIGITAL_LINES:
        page.insert_text((72, y), line, fontsize=18)
        y += 30
    doc.save(str(src))

    # Rasterise to an image-only PDF so there is no recoverable text layer.
    out = tmp_path / "scanned.pdf"
    img_doc = fitz.open()
    for p in doc:
        pix = p.get_pixmap(dpi=150)
        img_page = img_doc.new_page(width=pix.width, height=pix.height)
        img_page.insert_image(img_page.rect, pixmap=pix)
    img_doc.save(str(out))
    img_doc.close()
    doc.close()
    return out
