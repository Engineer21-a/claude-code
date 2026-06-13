"""Extract born-digital words + boxes (PyMuPDF) into a PageText.

Boxes are in PDF points. The pipeline converts to render pixels via
`pdf_renderer.scale_for_dpi` when burning Mode A boxes.
"""
from __future__ import annotations

from typing import List

import fitz  # PyMuPDF

from app.core.span_mapper import build_page_text
from app.detectors.base import Box, PageText, Token


def extract_page_text(page: "fitz.Page") -> PageText:
    """Return a PageText for a born-digital page (boxes in PDF points)."""
    # words: (x0, y0, x1, y1, "word", block_no, line_no, word_no), reading order.
    raw = page.get_text("words", sort=True)

    tokens: List[Token] = []
    for x0, y0, x1, y1, word, block_no, line_no, _word_no in raw:
        if not word:
            continue
        # A stable per-line id; PyMuPDF resets line_no within each block.
        line_id = block_no * 10_000 + line_no
        tokens.append(
            Token(
                text=word,
                start=0,  # filled by build_page_text
                end=0,
                box=Box(x0, y0, x1, y1),
                line_id=line_id,
            )
        )

    return build_page_text(tokens)
