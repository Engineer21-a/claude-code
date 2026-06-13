"""span_mapper: a known span must land on the correct token boxes."""
from __future__ import annotations

from app.core.span_mapper import build_page_text, map_detection_to_boxes
from app.detectors.base import Box, Detection, Token


def _toks():
    # Two lines: "Kundin: Erika Mustermann" / "IBAN: DE89"
    raw = [
        Token("Kundin:", 0, 0, Box(0, 0, 50, 10), line_id=0),
        Token("Erika", 0, 0, Box(55, 0, 90, 10), line_id=0),
        Token("Mustermann", 0, 0, Box(95, 0, 170, 10), line_id=0),
        Token("IBAN:", 0, 0, Box(0, 20, 40, 30), line_id=1),
        Token("DE89", 0, 0, Box(45, 20, 80, 30), line_id=1),
    ]
    return build_page_text(raw)


def test_offsets_assigned():
    page = _toks()
    assert page.text == "Kundin: Erika Mustermann\nIBAN: DE89"
    # The word "Erika" sits where we expect.
    assert page.text[page.tokens[1].start : page.tokens[1].end] == "Erika"


def test_span_maps_to_two_tokens_one_box():
    page = _toks()
    start = page.text.index("Erika")
    end = page.text.index("Mustermann") + len("Mustermann")
    det = Detection(start, end, "user_word", "Erika Mustermann")
    boxes = map_detection_to_boxes(page, det)
    assert len(boxes) == 1  # both tokens on one line -> one union box
    b = boxes[0]
    assert (b.x0, b.x1) == (55, 170)


def test_multiline_span_gives_box_per_line():
    page = _toks()
    det = Detection(0, len(page.text), "x", page.text)
    boxes = map_detection_to_boxes(page, det)
    assert len(boxes) == 2  # one box per text line


def test_full_line_expands():
    page = _toks()
    start = page.text.index("Erika")
    det = Detection(start, start + 5, "user_word", "Erika", full_line=True)
    boxes = map_detection_to_boxes(page, det)
    assert len(boxes) == 1
    # Full line covers from the first token (x0=0) to the last (x1=170).
    assert (boxes[0].x0, boxes[0].x1) == (0, 170)


def test_margin_expands_box():
    page = _toks()
    start = page.text.index("Erika")
    det = Detection(start, start + 5, "user_word", "Erika")
    boxes = map_detection_to_boxes(page, det, margin=4)
    assert boxes[0].x0 == 55 - 4
