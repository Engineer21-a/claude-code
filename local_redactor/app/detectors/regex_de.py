"""Layer 3 — German identifier regex (deterministic, high precision).

Each pattern is toggleable. Broad patterns (dates, PLZ) are gated by context so
they don't over-redact. Patterns run against the reconstructed page string;
`span_mapper` turns matches into boxes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import regex

from app.detectors.base import Detection, PageText

# Reusable context fragments. Broad patterns only fire near these.
_STREET_HINT = regex.compile(
    r"(?:stra(?:ß|ss)e|str\.|weg|allee|platz|gasse|ring|ufer)\s+\d{1,4}",
    regex.IGNORECASE,
)
_NAME_OR_LABEL_HINT = regex.compile(
    r"(?:geb(?:oren|\.)|geburtsdatum|geb\.-datum|am\s|name|herr|frau)",
    regex.IGNORECASE,
)


@dataclass
class DePattern:
    name: str
    pattern: "regex.Pattern"
    enabled: bool = True
    score: float = 1.0
    #: Optional gate: receives (match, full_text) and returns keep/drop.
    context_gate: Optional[Callable[["regex.Match", str], bool]] = None
    #: Opt-in (off by default) — Bundesland-specific or otherwise risky.
    opt_in: bool = False


def _near(text: str, start: int, end: int, hint: "regex.Pattern", window: int = 40) -> bool:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return hint.search(text, lo, hi) is not None


def _default_patterns() -> List[DePattern]:
    P = regex.compile
    I = regex.IGNORECASE
    return [
        DePattern("iban", P(r"\bDE(?:\s?\d){20}\b")),
        # 11-digit tax id (IdNr), optionally space-grouped.
        DePattern("steuer_id", P(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b")),
        # Steuernummer varies by Bundesland -> opt-in.
        DePattern("steuernummer", P(r"\b\d{2,3}/\d{3}/\d{4,5}\b"), opt_in=True),
        # Sozialversicherungsnummer: 2 digits, 6-digit DOB, 1 letter, 2 digits, 1 check digit.
        DePattern("sozialversicherung", P(r"\b\d{2}\d{6}[A-Z]\d{2}\d\b")),
        # eGK Krankenversichertennummer: letter + 9 digits.
        DePattern("egk", P(r"\b[A-Z]\d{9}\b")),
        DePattern("bic", P(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")),
        DePattern("credit_card", P(r"\b(?:\d[ -]?){13,19}\b"), score=0.9),
        # German phone numbers (+49 or 0 prefixes, varied grouping).
        DePattern("phone_de", P(r"(?:\+49|0)(?:[\s/\-]?\d){6,13}\b")),
        DePattern("email", P(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b")),
        # Dates DD.MM.YYYY — broad: gate near a name/label.
        DePattern(
            "date_numeric",
            P(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b"),
            context_gate=lambda m, t: _near(t, m.start(), m.end(), _NAME_OR_LABEL_HINT),
            score=0.7,
        ),
        # Textual German dates like "1. Juli 2026" — gate near a name/label.
        DePattern(
            "date_textual",
            P(
                r"\b\d{1,2}\.\s?(?:Januar|Februar|März|April|Mai|Juni|Juli|August|"
                r"September|Oktober|November|Dezember)\s+\d{4}\b",
                I,
            ),
            context_gate=lambda m, t: _near(t, m.start(), m.end(), _NAME_OR_LABEL_HINT),
            score=0.7,
        ),
        # PLZ: 5 digits — too broad alone, only near a street+number context.
        DePattern(
            "plz",
            P(r"\b\d{5}\b"),
            context_gate=lambda m, t: _near(t, m.start(), m.end(), _STREET_HINT, window=60),
            score=0.6,
        ),
        # Kfz-Kennzeichen.
        DePattern("kfz", P(r"\b[A-ZÄÖÜ]{1,3}-[A-Z]{1,2}\s?\d{1,4}\b")),
    ]


class RegexDeDetector:
    name = "regex_de"

    def __init__(self, patterns: Optional[List[DePattern]] = None, include_opt_in: bool = False):
        self._patterns = patterns or _default_patterns()
        self._include_opt_in = include_opt_in

    def detect(self, page: PageText) -> List[Detection]:
        text = page.text
        detections: List[Detection] = []
        for pat in self._patterns:
            if not pat.enabled:
                continue
            if pat.opt_in and not self._include_opt_in:
                continue
            for m in pat.pattern.finditer(text):
                if pat.context_gate and not pat.context_gate(m, text):
                    continue
                detections.append(
                    Detection(
                        start=m.start(),
                        end=m.end(),
                        label=f"regex:{pat.name}",
                        text=m.group(0),
                        score=pat.score,
                        source=self.name,
                    )
                )
        return detections
