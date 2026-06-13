"""OCR wrapper -> tokens with pixel boxes, for RapidOCR (default) and Tesseract.

Heavy OCR backends are imported lazily so the rest of the package imports and
tests run without them. Both engines yield `Token`s whose boxes are in image
device pixels (the same space Mode A paints in).
"""
from __future__ import annotations

from typing import List, Tuple

from PIL import Image

from app.config.settings import OcrEngine
from app.core.span_mapper import build_page_text
from app.detectors.base import Box, PageText, Token


# --------------------------------------------------------------------------- #
# Line grouping shared by both engines.
# --------------------------------------------------------------------------- #
def _assign_line_ids(words: List[Tuple[str, Box, float]]) -> List[Token]:
    """Group word boxes into reading-order lines by vertical overlap.

    `words` is (text, box, confidence) in arbitrary order. We sort top-to-bottom
    then left-to-right, and start a new line whenever a token's vertical centre
    drops below the current line's band.
    """
    if not words:
        return []

    items = sorted(words, key=lambda w: (w[1].y0, w[1].x0))
    tokens: List[Token] = []
    line_id = 0
    line_top = items[0][1].y0
    line_bottom = items[0][1].y1

    for text, box, conf in items:
        center = (box.y0 + box.y1) / 2.0
        if center > line_bottom:
            # Token starts below the current line band -> new line.
            line_id += 1
            line_top, line_bottom = box.y0, box.y1
        else:
            line_bottom = max(line_bottom, box.y1)
        tokens.append(
            Token(text=text, start=0, end=0, box=box, line_id=line_id, confidence=conf)
        )

    # Re-sort within each line left-to-right (sort was by y0 first).
    tokens.sort(key=lambda t: (t.line_id, t.box.x0))
    return tokens


# --------------------------------------------------------------------------- #
# Engines.
# --------------------------------------------------------------------------- #
class _RapidOcr:
    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            try:
                from rapidocr import RapidOCR  # newer consolidated package name
            except ImportError as exc:
                raise ImportError(
                    "RapidOCR not installed. `pip install rapidocr-onnxruntime`."
                ) from exc
        self._engine = RapidOCR()

    def recognise(self, img: Image.Image) -> List[Tuple[str, Box, float]]:
        import numpy as np

        result, _ = self._engine(np.array(img))
        words: List[Tuple[str, Box, float]] = []
        for entry in result or []:
            quad, text, conf = entry
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            box = Box(min(xs), min(ys), max(xs), max(ys))
            words.append((text, box, float(conf)))
        return words


class _Tesseract:
    def __init__(self, language: str = "deu") -> None:
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pytesseract not installed.") from exc
        self._pt = pytesseract
        self._lang = language

    def recognise(self, img: Image.Image) -> List[Tuple[str, Box, float]]:
        data = self._pt.image_to_data(
            img, lang=self._lang, output_type=self._pt.Output.DICT
        )
        words: List[Tuple[str, Box, float]] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:  # Tesseract uses -1 for non-text boxes.
                continue
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            words.append((text, Box(x, y, x + w, y + h), conf / 100.0))
        return words


def _make_engine(engine: OcrEngine, language: str):
    if engine == OcrEngine.TESSERACT:
        return _Tesseract(language)
    return _RapidOcr()


def ocr_image(
    img: Image.Image, engine: OcrEngine = OcrEngine.RAPIDOCR, language: str = "deu"
) -> PageText:
    """OCR a rendered page image into a PageText (pixel-space boxes)."""
    backend = _make_engine(engine, language)
    words = backend.recognise(img)
    tokens = _assign_line_ids(words)
    return build_page_text(tokens)
