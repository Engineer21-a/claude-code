"""Layer 2 — fuzzy OCR matching with rapidfuzz.

OCR confuses 1<->l<->I, 0<->O, rn<->m, so `Mu1ler` should still match `Müller`.
This runs over the page tokens (per-token and over adjacent token windows for
multi-word terms) and is kept separate from Layer 1 so it can be tuned or
disabled independently.
"""
from __future__ import annotations

from typing import List

from rapidfuzz import fuzz

from app.config.settings import UserWord
from app.detectors.base import Detection, PageText, Token


def _norm(s: str) -> str:
    # Collapse the common OCR confusions before comparing.
    table = str.maketrans({"1": "l", "I": "l", "|": "l", "0": "o"})
    return s.translate(table).casefold().replace("rn", "m")


class FuzzyDetector:
    name = "fuzzy"

    def __init__(self, words: List[UserWord], threshold: int = 85, max_window: int = 4):
        self._terms = [w.text for w in words if w.text.strip()]
        self._threshold = threshold
        self._max_window = max_window

    def detect(self, page: PageText) -> List[Detection]:
        if not self._terms:
            return []
        toks: List[Token] = page.tokens
        detections: List[Detection] = []

        for term in self._terms:
            n_words = max(1, len(term.split()))
            target = _norm(term)
            # Slide a window of up to n_words..max_window adjacent tokens.
            for size in range(n_words, min(self._max_window, len(toks)) + 1):
                for i in range(0, len(toks) - size + 1):
                    window = toks[i : i + size]
                    # Only join tokens on the same line for phrase matching.
                    if len({t.line_id for t in window}) > 1:
                        continue
                    candidate = _norm(" ".join(t.text for t in window))
                    score = fuzz.ratio(candidate, target)
                    if score >= self._threshold:
                        detections.append(
                            Detection(
                                start=window[0].start,
                                end=window[-1].end,
                                label="fuzzy",
                                text=" ".join(t.text for t in window),
                                score=score / 100.0,
                                source=self.name,
                            )
                        )
        return detections
