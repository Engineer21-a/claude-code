"""Typed application settings (pydantic) with JSON load/save.

Persists only preferences — never document text or detected PII (Hard Invariant).
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from .paths import settings_path


class OutputMode(str, Enum):
    #: Mode A — flatten every page to a burned-in image, no text layer at all.
    MAXIMUM_SAFE = "maximum_safe"
    #: Mode B — PyMuPDF stream redaction, born-digital pages only.
    STRUCTURE_PRESERVING = "structure_preserving"


class OcrEngine(str, Enum):
    RAPIDOCR = "rapidocr"
    TESSERACT = "tesseract"


class UserWord(BaseModel):
    """A single user-supplied term plus its matching options."""

    text: str
    case_insensitive: bool = True
    whole_word_only: bool = True
    normalize_umlauts: bool = True
    redact_full_line: bool = False
    redact_with_margin: bool = True
    #: Once a human confirms it, redact this term everywhere it appears.
    redact_everywhere: bool = True


class Settings(BaseModel):
    ocr_engine: OcrEngine = OcrEngine.RAPIDOCR
    ocr_language: str = "deu"
    dpi: int = 300
    redaction_margin_px: int = 4
    default_output_mode: OutputMode = OutputMode.MAXIMUM_SAFE
    keep_original_untouched: bool = True
    save_audit_report: bool = True

    enable_regex_de: bool = True
    enable_gliner: bool = True
    gliner_threshold: float = 0.35
    enable_fuzzy: bool = True
    enable_llm: bool = False

    user_words: List[UserWord] = Field(default_factory=list)
    fuzzy_threshold: int = 85

    #: GLiNER2 labels to request at inference time (subset of its 42 types).
    gliner_labels: List[str] = Field(
        default_factory=lambda: [
            "person",
            "address",
            "phone_number",
            "email",
            "date_of_birth",
            "bank_account",
            "health_insurance_id",
            "company",
            "government_id",
        ]
    )

    #: Optional explicit opt-in for the network update check. Default off.
    allow_update_check: bool = False

    # --- Phase 4 hardening ---
    #: Pages whose mean OCR confidence falls below this are warned about.
    ocr_min_confidence: float = 0.55
    #: When a scanned page reads poorly, re-render + re-OCR at this higher DPI.
    auto_retry_low_confidence: bool = True
    ocr_retry_dpi: int = 450
    #: Enabling debug logging risks document text in logs — off by default, warned.
    debug_logging: bool = False

    def save(self, path: Optional[Path] = None) -> None:
        path = path or settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        path = path or settings_path()
        if not path.exists():
            s = cls()
            s.save(path)
            return s
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            # Corrupt settings should never block the app; fall back to defaults
            # without clobbering the user's file silently.
            return cls()
