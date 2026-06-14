"""Orchestrate redaction of one file end to end.

Per page: classify -> extract text (born-digital) or OCR (scan) -> run enabled
detectors -> map spans to boxes -> burn (Mode A) or stream-redact (Mode B) ->
verify -> write output. The original file is opened read-only and never changed.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from app.config.settings import OcrEngine, OutputMode, Settings
from app.core import document_loader, page_classifier, pdf_renderer, redaction_engine
from app.core.audit import AuditReport, count_by_source
from app.core.document_loader import DocType
from app.core.output_paths import redacted_output_path
from app.core.secure_temp import warn_if_debug_logging
from app.core.span_mapper import map_detections_to_boxes
from app.core.text_extractor import extract_page_text
from app.core.verification import verify_output
from app.detectors.base import Box, Detection, PageText

logger = logging.getLogger("localredactor")


class EncryptedDocumentError(Exception):
    """Raised when an input PDF is password-protected and cannot be opened."""


class CorruptDocumentError(Exception):
    """Raised when an input file cannot be parsed."""


@dataclass
class PageResult:
    index: int
    kind: str
    detections: List[Detection] = field(default_factory=list)
    boxes: List[Box] = field(default_factory=list)


@dataclass
class FileResult:
    source: Path
    output: Optional[Path]
    audit: AuditReport
    success: bool


# A detector is anything with `.detect(PageText) -> List[Detection]`.
DetectorFn = Callable[[PageText], List[Detection]]


def _build_detectors(settings: Settings) -> List[DetectorFn]:
    """Instantiate the enabled text detectors in trust order.

    GLiNER2 and the optional LLM are added by the caller (Phase 3/5) since they
    carry heavy imports; this keeps Phase 1/2 importable without them.
    """
    detectors: List[DetectorFn] = []

    # Layer 1 — user words (highest trust).
    if settings.user_words:
        from app.detectors.word_detector import WordDetector

        detectors.append(WordDetector(settings.user_words).detect)

    # Layer 3 — German regex.
    if settings.enable_regex_de:
        from app.detectors.regex_de import RegexDeDetector

        detectors.append(RegexDeDetector().detect)

    # Layer 2 — fuzzy OCR matching (only meaningful with user words).
    if settings.enable_fuzzy and settings.user_words:
        from app.detectors.fuzzy_detector import FuzzyDetector

        detectors.append(
            FuzzyDetector(settings.user_words, threshold=settings.fuzzy_threshold).detect
        )

    # Layer 4 — GLiNER2-PII semantic detection (heavy; lazy model load).
    if settings.enable_gliner:
        from app.detectors.gliner_detector import GlinerDetector
        from app.config.paths import bundled_models_dir

        bundled = bundled_models_dir() / "gliner2-pii-v1"
        detectors.append(
            GlinerDetector(
                labels=settings.gliner_labels,
                threshold=settings.gliner_threshold,
                model_path=str(bundled) if bundled.exists() else None,
            ).detect
        )

    return detectors


def _mean_confidence(page_text: PageText) -> float:
    confs = [t.confidence for t in page_text.tokens if t.confidence >= 0]
    if not confs:
        return 1.0
    return sum(confs) / len(confs)


def _ocr_at(page: "fitz.Page", settings: Settings, dpi: int) -> PageText:
    from app.core.ocr_engine import ocr_image

    img = pdf_renderer.render_page(page, dpi=dpi)
    return ocr_image(img, settings.ocr_engine, settings.ocr_language)


def _page_text_for(
    page: "fitz.Page",
    kind: page_classifier.PageKind,
    settings: Settings,
    warnings: List[str],
    page_index: int,
) -> Tuple[PageText, str, int]:
    """Return (PageText, box_space, render_dpi) for a page.

    For scans, surface an OCR-confidence warning and (optionally) re-OCR at a
    higher DPI when the page reads poorly. The returned `render_dpi` is the DPI
    the boxes are in, so the final Mode A render uses the same DPI per page.
    """
    if kind == page_classifier.PageKind.BORN_DIGITAL:
        return extract_page_text(page), "points", settings.dpi

    # Scan: render + OCR at the configured DPI.
    page_text = _ocr_at(page, settings, settings.dpi)
    used_dpi = settings.dpi
    conf = _mean_confidence(page_text)

    if conf < settings.ocr_min_confidence:
        if settings.auto_retry_low_confidence and settings.dpi < settings.ocr_retry_dpi:
            retry = _ocr_at(page, settings, settings.ocr_retry_dpi)
            retry_conf = _mean_confidence(retry)
            warnings.append(
                f"page {page_index + 1}: low OCR confidence "
                f"({conf:.0%}); retried at {settings.ocr_retry_dpi} DPI "
                f"({retry_conf:.0%})."
            )
            if retry_conf >= conf:
                page_text, used_dpi = retry, settings.ocr_retry_dpi
        else:
            warnings.append(
                f"page {page_index + 1}: low OCR confidence "
                f"({conf:.0%}); consider a higher DPI."
            )
    return page_text, "pixels", used_dpi


def _run_detectors(page_text: PageText, detectors: List[DetectorFn]) -> List[Detection]:
    out: List[Detection] = []
    for fn in detectors:
        out.extend(fn(page_text))
    return out


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _open_readonly(
    doc_path: Path, doc_type: DocType, password: Optional[str] = None
) -> "fitz.Document":
    """Open input read-only, handling images, encryption, and corruption.

    Raises EncryptedDocumentError or CorruptDocumentError so the caller can flag
    the file without aborting the rest of the batch.
    """
    try:
        if doc_type == DocType.IMAGE:
            rect_doc = fitz.open(str(doc_path))  # opened read-only; never saved
            try:
                pdfbytes = rect_doc.convert_to_pdf()
            finally:
                rect_doc.close()
            return fitz.open(stream=pdfbytes, filetype="pdf")
        doc = fitz.open(str(doc_path))
    except Exception as exc:  # PyMuPDF raises various errors on bad input
        raise CorruptDocumentError(str(exc)) from exc

    # Encrypted PDFs: try an empty password, then a supplied one. The
    # `authenticate` return code (>0 == success) is authoritative — `needs_pass`
    # is not reliably cleared across PyMuPDF versions.
    if doc.needs_pass:
        code = doc.authenticate("")
        if not code and password is not None:
            code = doc.authenticate(password)
        if not code:
            doc.close()
            raise EncryptedDocumentError(
                "PDF is password-protected; supply the password to redact it."
            )
    return doc


def process_file(
    source: str | Path,
    settings: Settings,
    *,
    out_dir: Optional[Path] = None,
    extra_detectors: Optional[List[DetectorFn]] = None,
    confirmed_boxes: Optional[Dict[int, List[Box]]] = None,
    ocr_verify: bool = False,
    password: Optional[str] = None,
) -> FileResult:
    """Redact one file, never raising — failures return a flagged FileResult.

    Encrypted, corrupt, and mixed PDFs are handled per file so a batch never
    aborts on a single bad input. `confirmed_boxes` (from the review UI) override
    detection; `extra_detectors` lets callers inject GLiNER2/LLM detectors.
    """
    # Loading + hashing can fail (missing/unreadable file): handle gracefully.
    try:
        loaded = document_loader.load(source)
    except (FileNotFoundError, OSError) as exc:
        audit = AuditReport(source_name=Path(source).name, source_sha256="", status="error", error=str(exc))
        return FileResult(Path(source), None, audit, success=False)

    audit = AuditReport(
        source_name=loaded.path.name,
        source_sha256=loaded.sha256,
        mode=settings.default_output_mode.value,
    )
    audit.warnings.extend(warn_if_debug_logging(settings.debug_logging))

    detectors = _build_detectors(settings)
    if extra_detectors:
        detectors.extend(extra_detectors)

    try:
        doc = _open_readonly(loaded.path, loaded.doc_type, password=password)
    except EncryptedDocumentError as exc:
        audit.status = "encrypted"
        audit.error = str(exc)
        logger.warning("%s: encrypted, skipped", loaded.path.name)
        return FileResult(loaded.path, None, audit, success=False)
    except CorruptDocumentError as exc:
        audit.status = "corrupt"
        audit.error = str(exc)
        logger.warning("%s: could not be parsed, skipped", loaded.path.name)
        return FileResult(loaded.path, None, audit, success=False)

    try:
        return _redact_open_document(
            doc, loaded, settings, audit, detectors, out_dir, confirmed_boxes, ocr_verify
        )
    except Exception as exc:  # any unexpected failure flags this file only
        audit.status = "error"
        audit.error = f"{type(exc).__name__}: {exc}"
        logger.exception("%s: processing error", loaded.path.name)
        return FileResult(loaded.path, None, audit, success=False)
    finally:
        doc.close()


def _redact_open_document(
    doc: "fitz.Document",
    loaded: document_loader.LoadedDocument,
    settings: Settings,
    audit: AuditReport,
    detectors: List[DetectorFn],
    out_dir: Optional[Path],
    confirmed_boxes: Optional[Dict[int, List[Box]]],
    ocr_verify: bool,
) -> FileResult:
    # Decide effective mode: Mode B requires a pure born-digital PDF; a mixed or
    # scanned document falls back to Mode A for the whole document.
    mode = settings.default_output_mode
    if mode == OutputMode.STRUCTURE_PRESERVING and not (
        loaded.doc_type == DocType.PDF
        and page_classifier.document_is_pure_born_digital(doc)
    ):
        mode = OutputMode.MAXIMUM_SAFE
        audit.warnings.append("Mode B unavailable for this document; used Mode A.")
    audit.mode = mode.value
    audit.pages = doc.page_count

    all_labels: List[str] = []
    forbidden_terms: List[str] = []
    page_results: List[PageResult] = []
    pixel_pages: List[List[Box]] = []
    page_dpis: List[int] = []          # effective render DPI per page (Mode A)
    point_pages: Dict[int, List[Box]] = {}

    for i, page in enumerate(doc):
        kind = page_classifier.classify_page(page)
        page_text, space, page_dpi = _page_text_for(page, kind, settings, audit.warnings, i)
        detections = _run_detectors(page_text, detectors)

        if confirmed_boxes is not None:
            boxes = confirmed_boxes.get(i, [])
        else:
            boxes = map_detections_to_boxes(
                page_text, detections, margin=settings.redaction_margin_px
            )

        all_labels.extend(d.source or d.label for d in detections)
        forbidden_terms.extend(d.text for d in detections)
        page_results.append(PageResult(i, kind.value, detections, boxes))

        if mode == OutputMode.MAXIMUM_SAFE:
            pixel_pages.append(redaction_engine.to_pixel_boxes(boxes, space, page_dpi))
            page_dpis.append(page_dpi)
        else:
            point_pages[i] = boxes  # born-digital -> already PDF points

    # ----- Build output -----
    if mode == OutputMode.MAXIMUM_SAFE:
        images = []
        for i, page in enumerate(doc):
            img = pdf_renderer.render_page(page, dpi=page_dpis[i])
            images.append(redaction_engine.burn_boxes_on_image(img, pixel_pages[i]))
        raw = redaction_engine.build_image_only_pdf(images)
        output_bytes = redaction_engine.strip_pdf_metadata_bytes(raw)
    else:
        output_bytes = redaction_engine.apply_mode_b(doc, point_pages)

    # ----- Verify (mandatory gate) -----
    result = verify_output(
        output_bytes, mode, forbidden_terms, ocr_check=ocr_verify, ocr_language=settings.ocr_language
    )
    audit.verification_passed = result.passed
    audit.verification_reasons = result.reasons
    audit.boxes_total = sum(len(p.boxes) for p in page_results)
    audit.boxes_by_source = count_by_source(all_labels)

    if not result.passed:
        # Discard output; original stays untouched; flag for manual review.
        audit.status = "flagged"
        return FileResult(loaded.path, None, audit, success=False)

    out_path = redacted_output_path(loaded.path, out_dir)
    out_path.write_bytes(output_bytes)
    audit.output_name = out_path.name
    audit.output_sha256 = _sha256_bytes(output_bytes)
    audit.status = "ok"

    if settings.save_audit_report:
        audit.save(out_path.with_suffix(".audit.json"))

    return FileResult(loaded.path, out_path, audit, success=True)
