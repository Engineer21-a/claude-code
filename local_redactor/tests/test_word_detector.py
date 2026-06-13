"""Layer 1 word detector: options incl. umlaut normalisation."""
from __future__ import annotations

from app.config.profiles import PRESETS, get_preset
from app.config.settings import UserWord
from app.core.span_mapper import build_page_text
from app.detectors.base import Token
from app.detectors.word_detector import WordDetector


def _page(text: str):
    toks, cur = [], 0
    for w in text.split(" "):
        toks.append(Token(w, 0, 0, box=None, line_id=0))  # type: ignore[arg-type]
        cur += len(w) + 1
    return build_page_text(toks, word_sep=" ")


def test_exact_whole_word():
    det = WordDetector([UserWord(text="Mustermann")])
    hits = det.detect(_page("Kundin Mustermann zahlt"))
    assert [h.text for h in hits] == ["Mustermann"]


def test_whole_word_blocks_substring():
    det = WordDetector([UserWord(text="Mann", whole_word_only=True)])
    assert det.detect(_page("Mustermann ist da")) == []


def test_umlaut_normalisation_both_directions():
    # Term written with umlaut matches the ae-spelling and vice versa.
    det = WordDetector([UserWord(text="Müller", normalize_umlauts=True, whole_word_only=False)])
    assert det.detect(_page("Herr Mueller kommt"))
    det2 = WordDetector([UserWord(text="Mueller", normalize_umlauts=True, whole_word_only=False)])
    assert det2.detect(_page("Herr Müller kommt"))


def test_case_insensitive():
    det = WordDetector([UserWord(text="erika", case_insensitive=True)])
    assert det.detect(_page("Frau ERIKA Mustermann"))


def test_full_line_flag_propagates():
    det = WordDetector([UserWord(text="Erika", redact_full_line=True)])
    hits = det.detect(_page("Frau Erika Mustermann"))
    assert hits and hits[0].full_line is True


def test_presets_are_well_formed():
    assert "default_de" in PRESETS
    p = get_preset("medical")
    assert "health_insurance_id" in p.gliner_labels
    # Every preset has a name and description.
    assert all(pr.name and pr.description for pr in PRESETS.values())
