"""Classify each PDF page: born-digital text vs image-only scan.

The distinction decides whether we can extract word boxes directly (born-digital)
or must OCR a rendered image (scan), and whether Mode B is even offered.
"""
from __future__ import annotations

from enum import Enum

import fitz  # PyMuPDF


class PageKind(str, Enum):
    BORN_DIGITAL = "born_digital"
    SCAN = "scan"


#: Pages with fewer than this many extractable word characters are treated as
#: scans (e.g. a digital wrapper around a single scanned image).
_MIN_TEXT_CHARS = 8


def classify_page(page: "fitz.Page") -> PageKind:
    words = page.get_text("words")
    char_count = sum(len(w[4]) for w in words)
    if char_count >= _MIN_TEXT_CHARS:
        return PageKind.BORN_DIGITAL
    return PageKind.SCAN


def document_is_pure_born_digital(doc: "fitz.Document") -> bool:
    """True only if every page is born-digital — gates Mode B eligibility."""
    return all(classify_page(p) == PageKind.BORN_DIGITAL for p in doc)
