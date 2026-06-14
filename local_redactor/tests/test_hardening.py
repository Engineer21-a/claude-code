"""Phase 4: encrypted/corrupt/missing inputs are handled per file, batch-safe."""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.config.settings import OutputMode, Settings, UserWord
from app.core.pipeline import process_file
from app.core.secure_temp import secure_workdir, warn_if_debug_logging
from tests.conftest import BORN_DIGITAL_LINES, SECRET_NAME


def _settings() -> Settings:
    return Settings(
        default_output_mode=OutputMode.MAXIMUM_SAFE,
        enable_gliner=False,
        enable_fuzzy=False,
        dpi=120,
        user_words=[UserWord(text=SECRET_NAME, whole_word_only=False)],
    )


def test_corrupt_pdf_is_flagged_not_raised(tmp_path: Path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf \x00\x01\x02")
    result = process_file(bad, _settings(), out_dir=tmp_path)
    assert result.success is False
    assert result.output is None
    assert result.audit.status == "corrupt"


def test_encrypted_pdf_is_flagged(tmp_path: Path):
    src = tmp_path / "secret.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), f"Kundin: {SECRET_NAME}", fontsize=12)
    doc.save(
        str(src),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="geheim",
        owner_pw="geheim",
    )
    doc.close()

    result = process_file(src, _settings(), out_dir=tmp_path)
    assert result.success is False
    assert result.audit.status == "encrypted"

    # With the right password it goes through and verifies clean.
    ok = process_file(src, _settings(), out_dir=tmp_path, password="geheim")
    assert ok.success is True
    assert ok.audit.verification_passed
    # Original untouched.
    assert src.exists()


def test_missing_file_is_flagged(tmp_path: Path):
    result = process_file(tmp_path / "nope.pdf", _settings(), out_dir=tmp_path)
    assert result.success is False
    assert result.audit.status == "error"


def test_batch_continues_past_bad_file(tmp_path: Path, born_digital_pdf: Path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 garbage")
    results = [process_file(p, _settings(), out_dir=tmp_path) for p in (bad, born_digital_pdf)]
    assert results[0].success is False
    assert results[1].success is True  # the good file still processed


def test_secure_workdir_wipes(tmp_path: Path):
    captured: Path
    with secure_workdir() as wd:
        captured = wd
        (wd / "scratch.bin").write_bytes(b"sensitive scratch data")
        assert (wd / "scratch.bin").exists()
    # Directory and its contents are gone after the context exits.
    assert not captured.exists()


def test_warn_if_debug_logging():
    warnings = warn_if_debug_logging(True)
    assert warnings and "Debug logging" in warnings[0]
