"""Phase 4: low OCR confidence is surfaced and (optionally) retried at higher DPI."""
from __future__ import annotations

from typing import List

import app.core.pipeline as pipeline
from app.config.settings import Settings
from app.core.page_classifier import PageKind
from app.core.span_mapper import build_page_text
from app.detectors.base import Box, Token


def _page_text(conf: float):
    return build_page_text([Token("Mustermann", 0, 0, Box(0, 0, 10, 10), 0, confidence=conf)])


def test_low_confidence_triggers_retry_at_higher_dpi(monkeypatch):
    calls: List[int] = []

    def fake_ocr_at(page, settings, dpi):
        calls.append(dpi)
        # First pass (low DPI) reads poorly; retry at higher DPI reads well.
        return _page_text(0.30 if dpi == settings.dpi else 0.92)

    monkeypatch.setattr(pipeline, "_ocr_at", fake_ocr_at)
    settings = Settings(dpi=300, ocr_retry_dpi=450, auto_retry_low_confidence=True, ocr_min_confidence=0.55)
    warnings: List[str] = []

    page_text, space, used_dpi = pipeline._page_text_for(None, PageKind.SCAN, settings, warnings, 0)

    assert calls == [300, 450]          # retried at the higher DPI
    assert used_dpi == 450              # kept the better result's DPI
    assert space == "pixels"
    assert warnings and "low OCR confidence" in warnings[0]


def test_low_confidence_warns_without_retry_when_disabled(monkeypatch):
    monkeypatch.setattr(pipeline, "_ocr_at", lambda page, settings, dpi: _page_text(0.20))
    settings = Settings(dpi=300, auto_retry_low_confidence=False, ocr_min_confidence=0.55)
    warnings: List[str] = []

    _, _, used_dpi = pipeline._page_text_for(None, PageKind.SCAN, settings, warnings, 2)

    assert used_dpi == 300
    assert warnings and "page 3" in warnings[0]  # 1-based page number in message


def test_good_confidence_no_warning(monkeypatch):
    monkeypatch.setattr(pipeline, "_ocr_at", lambda page, settings, dpi: _page_text(0.95))
    settings = Settings(dpi=300, ocr_min_confidence=0.55)
    warnings: List[str] = []
    pipeline._page_text_for(None, PageKind.SCAN, settings, warnings, 0)
    assert warnings == []
