"""GLiNER sliding-window chunking + cross-window span dedup (no model needed)."""
from __future__ import annotations

from app.detectors.base import Detection
from app.detectors.gliner_detector import chunk_text, dedupe_spans


def test_short_text_is_single_window():
    wins = chunk_text("Erika Mustermann wohnt in Berlin.")
    assert len(wins) == 1
    assert wins[0].start == 0


def test_long_text_windows_overlap_and_cover():
    text = " ".join(f"wort{i}" for i in range(1000))  # well over the window size
    wins = chunk_text(text, window_chars=200, overlap_chars=50)
    assert len(wins) > 1
    # Windows cover the whole string and carry correct base offsets.
    for w in wins:
        assert text[w.start : w.start + len(w.text)] == w.text
    assert wins[-1].start + len(wins[-1].text) == len(text)
    # Consecutive windows overlap.
    assert wins[1].start < wins[0].start + len(wins[0].text)


def test_dedupe_keeps_highest_confidence_for_overlap():
    dets = [
        Detection(0, 16, "gliner:person", "Erika Mustermann", score=0.6),
        Detection(0, 16, "gliner:person", "Erika Mustermann", score=0.9),  # dup, higher
        Detection(40, 46, "gliner:person", "Berlin", score=0.7),           # distinct
    ]
    out = dedupe_spans(dets)
    assert len(out) == 2
    person = [d for d in out if d.start == 0][0]
    assert person.score == 0.9


def test_dedupe_separates_different_labels():
    dets = [
        Detection(0, 5, "gliner:person", "Erika", score=0.8),
        Detection(0, 5, "gliner:address", "Erika", score=0.8),
    ]
    assert len(dedupe_spans(dets)) == 2
