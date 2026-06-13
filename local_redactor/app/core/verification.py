"""The mandatory leak-check gate (Hard Invariant).

Run after every export. If any check fails, the caller discards the output and
flags the file for manual review — the original is never touched regardless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF

from app.config.settings import OutputMode


@dataclass
class VerificationResult:
    passed: bool
    checks: dict = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def fail(self, check: str, reason: str) -> None:
        self.checks[check] = False
        self.reasons.append(reason)
        self.passed = False

    def ok(self, check: str) -> None:
        self.checks.setdefault(check, True)


def _normalise(s: str) -> str:
    # Loose comparison so spacing/case differences don't hide a leak.
    return "".join(s.split()).casefold()


def verify_output(
    output_bytes: bytes,
    mode: OutputMode,
    forbidden_terms: List[str],
    *,
    ocr_check: bool = False,
    ocr_language: str = "deu",
) -> VerificationResult:
    """Verify a redacted PDF blob against the four required checks."""
    result = VerificationResult(passed=True)
    forbidden = [t for t in forbidden_terms if t and t.strip()]
    norm_forbidden = [_normalise(t) for t in forbidden]

    doc = fitz.open(stream=output_bytes, filetype="pdf")
    try:
        # 1. Text-extraction check.
        extracted = "".join(page.get_text("text") for page in doc)
        if mode == OutputMode.MAXIMUM_SAFE:
            if extracted.strip():
                result.fail(
                    "text_extraction",
                    "Mode A output still has an extractable text layer.",
                )
            else:
                result.ok("text_extraction")
        else:  # Mode B: confirmed terms must not reappear.
            norm_extracted = _normalise(extracted)
            leaked = [t for t, n in zip(forbidden, norm_forbidden) if n and n in norm_extracted]
            if leaked:
                result.fail("text_extraction", f"Redacted term(s) still extractable: {len(leaked)} found.")
            else:
                result.ok("text_extraction")

        # 2. Raw-byte search (catches leftover streams / metadata).
        raw = output_bytes
        byte_leaks = 0
        for term in forbidden:
            for variant in {term, term.replace(" ", "")}:
                if variant and variant.encode("utf-8") in raw:
                    byte_leaks += 1
                    break
        if byte_leaks:
            result.fail("raw_bytes", f"{byte_leaks} forbidden string(s) found in raw bytes.")
        else:
            result.ok("raw_bytes")

        # 3. Metadata check.
        meta = doc.metadata or {}
        dirty = [k for k in ("title", "author", "subject", "keywords", "producer") if (meta.get(k) or "").strip()]
        xml = None
        try:
            xml = doc.xref_xml_metadata()
        except Exception:
            xml = None
        if dirty or xml:
            result.fail("metadata", f"Residual metadata: {dirty}{' +XMP' if xml else ''}.")
        else:
            result.ok("metadata")

        # 4. Optional OCR leak check (slow; opt-in).
        if ocr_check and forbidden:
            leak = _ocr_leak(doc, norm_forbidden, ocr_language)
            if leak:
                result.fail("ocr_leak", "OCR could still read a redacted term in the output.")
            else:
                result.ok("ocr_leak")
    finally:
        doc.close()

    return result


def _ocr_leak(doc: "fitz.Document", norm_forbidden: List[str], language: str) -> bool:
    try:
        from PIL import Image

        from app.config.settings import OcrEngine
        from app.core.ocr_engine import ocr_image
    except ImportError:  # pragma: no cover - OCR optional
        return False

    for page in doc:
        pix = page.get_pixmap(dpi=200, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        try:
            page_text = ocr_image(img, OcrEngine.TESSERACT, language)
        except ImportError:  # pragma: no cover
            return False
        norm = _normalise(page_text.text)
        if any(n and n in norm for n in norm_forbidden):
            return True
    return False
