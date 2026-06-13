"""German regex library: precision + context gating."""
from __future__ import annotations

from app.core.span_mapper import build_page_text
from app.detectors.base import Token
from app.detectors.regex_de import RegexDeDetector


def _page(text: str):
    # One token per whitespace chunk is enough for regex-over-text tests.
    toks = []
    cursor = 0
    for word in text.split(" "):
        toks.append(Token(word, 0, 0, box=None, line_id=0))  # type: ignore[arg-type]
        cursor += len(word) + 1
    page = build_page_text(toks, word_sep=" ")
    return page


def _labels(text: str, include_opt_in: bool = False):
    det = RegexDeDetector(include_opt_in=include_opt_in)
    return {d.label: d.text for d in det.detect(_page(text))}


def test_iban_detected():
    out = _labels("IBAN: DE89 3704 0044 0532 0130 00 danke")
    assert "regex:iban" in out


def test_email_detected():
    out = _labels("Mail erika.mustermann@example.de senden")
    assert out.get("regex:email") == "erika.mustermann@example.de"


def test_phone_detected():
    out = _labels("Tel +49 170 1234567 erreichbar")
    assert "regex:phone_de" in out


def test_plain_5_digits_not_redacted_without_street_context():
    # A bare 5-digit run must NOT be flagged as a PLZ.
    out = _labels("Bestellung 12345 wurde versandt")
    assert "regex:plz" not in out


def test_plz_redacted_with_street_context():
    out = _labels("Hauptstraße 5 12345 Berlin")
    assert "regex:plz" in out


def test_date_gated_by_name_context():
    assert "regex:date_numeric" not in _labels("Version 1.2.2026 build")
    assert "regex:date_numeric" in _labels("geboren am 01.07.2026 in Berlin")


def test_steuernummer_is_opt_in():
    text = "Steuernummer 21/815/08150 angeben"
    assert "regex:steuernummer" not in _labels(text)
    assert "regex:steuernummer" in _labels(text, include_opt_in=True)
