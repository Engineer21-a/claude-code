"""Hard Invariant: a redacted export must contain no recoverable sensitive data.

Encodes the text-extraction and raw-byte checks against the sample fixtures,
for both born-digital and scanned input, in both output modes. Runs in CI.
"""
from __future__ import annotations

from pathlib import Path

from app.config.settings import OcrEngine, OutputMode, Settings, UserWord
from app.core import document_loader
from app.core.pipeline import process_file
from tests.conftest import SECRET_EMAIL, SECRET_NAME


def _settings(mode: OutputMode) -> Settings:
    # Keep the test self-contained and OCR-free: drive redaction from the word
    # list and regex, both of which work on the born-digital text layer.
    return Settings(
        default_output_mode=mode,
        enable_gliner=False,
        enable_fuzzy=False,
        enable_regex_de=True,
        dpi=150,
        user_words=[UserWord(text=SECRET_NAME, whole_word_only=False)],
    )


def test_mode_a_born_digital_no_text_layer(born_digital_pdf: Path, tmp_path: Path):
    before = document_loader.load(born_digital_pdf)
    result = process_file(born_digital_pdf, _settings(OutputMode.MAXIMUM_SAFE), out_dir=tmp_path)

    assert result.success
    assert result.output is not None
    assert result.audit.verification_passed

    # Original untouched.
    after = document_loader.load(born_digital_pdf)
    assert before.sha256 == after.sha256

    # Output has NO extractable text at all (Mode A).
    out_bytes = result.output.read_bytes()
    import fitz

    doc = fitz.open(stream=out_bytes, filetype="pdf")
    extracted = "".join(p.get_text("text") for p in doc)
    doc.close()
    assert extracted.strip() == ""

    # Raw bytes carry none of the secrets.
    assert SECRET_NAME.encode() not in out_bytes
    assert SECRET_EMAIL.encode() not in out_bytes


def test_mode_b_born_digital_terms_removed(born_digital_pdf: Path, tmp_path: Path):
    result = process_file(
        born_digital_pdf, _settings(OutputMode.STRUCTURE_PRESERVING), out_dir=tmp_path
    )
    assert result.success
    assert result.audit.mode == OutputMode.STRUCTURE_PRESERVING.value

    out_bytes = result.output.read_bytes()
    import fitz

    doc = fitz.open(stream=out_bytes, filetype="pdf")
    extracted = "".join(p.get_text("text") for p in doc)
    doc.close()

    # The redacted name and email must not reappear; other text may remain.
    assert SECRET_NAME not in extracted
    assert SECRET_EMAIL not in extracted
    assert SECRET_NAME.encode() not in out_bytes


def test_mode_b_falls_back_to_a_for_scans(scanned_pdf: Path, tmp_path: Path):
    # A scanned doc requested in Mode B must fall back to Mode A automatically.
    settings = _settings(OutputMode.STRUCTURE_PRESERVING)
    # No OCR available in CI -> drive with regex only (still produces a valid,
    # empty-text Mode A output even with zero boxes).
    settings.user_words = []
    settings.enable_fuzzy = False
    # Prefer RapidOCR if installed; fall back/skip handled below.
    settings.ocr_engine = OcrEngine.RAPIDOCR
    try:
        result = process_file(scanned_pdf, settings, out_dir=tmp_path)
    except ImportError:
        # OCR backend not installed in this environment; skip gracefully.
        import pytest

        pytest.skip("OCR backend not installed")
    assert result.audit.mode == OutputMode.MAXIMUM_SAFE
    assert result.success
    out_bytes = result.output.read_bytes()
    import fitz

    doc = fitz.open(stream=out_bytes, filetype="pdf")
    extracted = "".join(p.get_text("text") for p in doc)
    doc.close()
    assert extracted.strip() == ""
