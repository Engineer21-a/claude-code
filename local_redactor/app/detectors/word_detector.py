"""Layer 1 — user words and phrases.

Per-term options: case-insensitive, whole-word-only, umlaut normalization
(ä<->ae, ö<->oe, ü<->ue, ß<->ss), and redact-full-line. Matching runs against
the reconstructed page string so it works identically for born-digital and
OCR'd pages; `span_mapper` turns the spans into boxes.
"""
from __future__ import annotations

from typing import List

import regex

from app.config.settings import UserWord
from app.detectors.base import Detection, PageText

_UMLAUT_EXPANSIONS = {
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
}
# After expansion, each ascii digraph may also match its single umlaut char.
_DIGRAPH_ALTERNATIVES = {
    "ae": "(?:ä|ae)", "oe": "(?:ö|oe)", "ue": "(?:ü|ue)", "ss": "(?:ß|ss)",
}


def _build_pattern(word: UserWord) -> "regex.Pattern":
    term = word.text
    if word.normalize_umlauts:
        for k, v in _UMLAUT_EXPANSIONS.items():
            term = term.replace(k, v)

    escaped = regex.escape(term)
    if word.normalize_umlauts:
        for digraph, alt in _DIGRAPH_ALTERNATIVES.items():
            escaped = escaped.replace(digraph, alt)

    if word.whole_word_only:
        escaped = rf"\b{escaped}\b"

    flags = regex.UNICODE
    if word.case_insensitive:
        flags |= regex.IGNORECASE
    return regex.compile(escaped, flags)


class WordDetector:
    name = "word"

    def __init__(self, words: List[UserWord]):
        self._words = [w for w in words if w.text.strip()]
        self._compiled = [(w, _build_pattern(w)) for w in self._words]

    def detect(self, page: PageText) -> List[Detection]:
        detections: List[Detection] = []
        for word, pattern in self._compiled:
            for m in pattern.finditer(page.text):
                detections.append(
                    Detection(
                        start=m.start(),
                        end=m.end(),
                        label="user_word",
                        text=m.group(0),
                        score=1.0,
                        source=self.name,
                        full_line=word.redact_full_line,
                    )
                )
        return detections
