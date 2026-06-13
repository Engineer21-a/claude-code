"""Map character spans <-> token boxes.

Every text detector (word list, regex, GLiNER2) runs against one reconstructed
page string. `span_mapper` is the single place that:

  1. builds that string from tokens while recording each token's char range, and
  2. translates a detected `[start, end)` span back into redaction boxes.

Get this right once and reuse it for every detector. There is a focused unit
test in `tests/test_ocr_boxes.py`.
"""
from __future__ import annotations

from typing import Dict, List

from app.detectors.base import Box, Detection, PageText, Token


def build_page_text(tokens: List[Token], *, line_sep: str = "\n", word_sep: str = " ") -> PageText:
    """Concatenate tokens in reading order, assigning each its char offsets.

    Tokens sharing a `line_id` are joined with `word_sep`; lines are joined with
    `line_sep`. The returned tokens are *copies* with `start`/`end` filled in so
    detectors and the mapper agree on offsets exactly.
    """
    placed: List[Token] = []
    parts: List[str] = []
    cursor = 0
    prev_line: int | None = None

    for tok in tokens:
        if prev_line is not None:
            sep = line_sep if tok.line_id != prev_line else word_sep
            parts.append(sep)
            cursor += len(sep)
        start = cursor
        parts.append(tok.text)
        cursor += len(tok.text)
        placed.append(
            Token(
                text=tok.text,
                start=start,
                end=cursor,
                box=tok.box,
                line_id=tok.line_id,
                confidence=tok.confidence,
            )
        )
        prev_line = tok.line_id

    return PageText(text="".join(parts), tokens=placed)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    # Half-open intervals; a zero-width detection still hits the token it lands in.
    if a_start == a_end:
        return b_start <= a_start < b_end
    return a_start < b_end and b_start < a_end


def map_detection_to_boxes(
    page: PageText, det: Detection, *, margin: float = 0.0
) -> List[Box]:
    """Return one box per text line touched by `det` (expanded by `margin`)."""
    touched = [
        tok for tok in page.tokens if _overlaps(det.start, det.end, tok.start, tok.end)
    ]
    if not touched:
        return []

    if det.full_line:
        # Expand to every token sharing a line with any touched token.
        line_ids = {t.line_id for t in touched}
        touched = [t for t in page.tokens if t.line_id in line_ids]

    # Union per line so a multi-line span produces a box per line, not one giant
    # rectangle swallowing the whitespace between lines.
    per_line: Dict[int, Box] = {}
    for tok in touched:
        if tok.line_id in per_line:
            per_line[tok.line_id] = per_line[tok.line_id].union(tok.box)
        else:
            per_line[tok.line_id] = tok.box

    boxes = list(per_line.values())
    if margin:
        boxes = [b.expand(margin) for b in boxes]
    return boxes


def map_detections_to_boxes(
    page: PageText, detections: List[Detection], *, margin: float = 0.0
) -> List[Box]:
    """Map and merge a whole detection list into a deduplicated box list."""
    boxes: List[Box] = []
    for det in detections:
        boxes.extend(map_detection_to_boxes(page, det, margin=margin))
    return merge_boxes(boxes)


def merge_boxes(boxes: List[Box]) -> List[Box]:
    """Greedily union overlapping boxes so we don't paint the same area twice."""
    remaining = list(boxes)
    merged: List[Box] = []
    while remaining:
        cur = remaining.pop()
        changed = True
        while changed:
            changed = False
            keep: List[Box] = []
            for other in remaining:
                if cur.intersects(other):
                    cur = cur.union(other)
                    changed = True
                else:
                    keep.append(other)
            remaining = keep
        merged.append(cur)
    return merged
