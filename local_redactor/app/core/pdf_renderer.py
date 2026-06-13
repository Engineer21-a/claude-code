"""Render a PDF page (or load an image) to a PIL.Image at a chosen DPI."""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


def scale_for_dpi(dpi: int) -> float:
    """Points-to-pixels factor. PDF user space is 72 points per inch."""
    return dpi / 72.0


def render_page(page: "fitz.Page", dpi: int = 300) -> Image.Image:
    """Rasterise one PDF page to an RGB PIL image at `dpi`."""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img.convert("RGB")


def load_image(path: str | Path) -> Image.Image:
    """Open a standalone image file read-only as RGB."""
    with Image.open(path) as im:
        return im.convert("RGB")
