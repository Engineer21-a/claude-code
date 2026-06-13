"""Hard Invariant: a pass over an input must never modify the input file."""
from __future__ import annotations

from pathlib import Path

from app.core import document_loader
from app.core.output_paths import redacted_output_path


def test_hash_stable_across_noop(born_digital_pdf: Path):
    before = document_loader.load(born_digital_pdf)
    # A "no-op pass": loading + hashing must not touch the bytes.
    after = document_loader.load(born_digital_pdf)
    assert before.sha256 == after.sha256
    assert before.size_bytes == after.size_bytes


def test_type_detection(born_digital_pdf: Path, scanned_pdf: Path):
    assert document_loader.load(born_digital_pdf).doc_type == document_loader.DocType.PDF
    assert document_loader.load(scanned_pdf).doc_type == document_loader.DocType.PDF


def test_output_path_never_equals_input(born_digital_pdf: Path, tmp_path: Path):
    out = redacted_output_path(born_digital_pdf, tmp_path)
    assert out != born_digital_pdf
    assert out.name.endswith("_redacted.pdf")
    assert not out.exists()  # computed, not yet written


def test_output_path_no_overwrite(tmp_path: Path):
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4\n")
    first = redacted_output_path(src, tmp_path)
    first.write_bytes(b"x")
    second = redacted_output_path(src, tmp_path)
    assert first != second
    assert second.name == "doc_redacted_1.pdf"
