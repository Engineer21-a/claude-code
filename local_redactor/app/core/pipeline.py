"""Orchestrate redaction of one file end to end.

Per page: classify -> extract text (born-digital) or OCR (scan) -> run enabled
detectors -> map spans to boxes -> burn (Mode A) or stream-redact (Mode B) ->
verify -> write output. The original file is opened read-only and never changed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import fitz  # PyMuPDF

from app.config.settings import OcrEngine, OutputMode, Settings
from app.core import document_loader, page_classifier, pdf_renderer, redaction_engine
from app.core.audit import AuditReport, count_by_source
from app.core.document_loader import DocType
from app.core.output_paths import redacted_output_path
from app.core.span_mapper import map_detections_to_boxes
from app.core.text_extractor import extract_page_text
from app.core.verification import verify_output
from app.detectors.base import Box, Detection, PageText


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


def _page_text_for(
    page: "fitz.Page", kind: page_classifier.PageKind, settings: Settings
) -> tuple[PageText, str]:
    """Return (PageText, box_space) for a page."""
    if kind == page_classifier.PageKind.BORN_DIGITAL:
        return extract_page_text(page), "points"
    # Scan: render + OCR. Boxes are in pixel space at the render DPI.
    from app.core.ocr_engine import ocr_image

    img = pdf_renderer.render_page(page, dpi=settings.dpi)
    return ocr_image(img, settings.ocr_engine, settings.ocr_language), "pixels"


def _run_detectors(page_text: PageText, detectors: List[DetectorFn]) -> List[Detection]:
    out: List[Detection] = []
    for fn in detectors:
        out.extend(fn(page_text))
    return out


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _open_readonly(doc_path: Path, doc_type: DocType) -> "fitz.Document":
    """Open input read-only. Images are wrapped into a one-page PDF in memory."""
    if doc_type == DocType.IMAGE:
        img_pdf = fitz.open()
        rect_doc = fitz.open(str(doc_path))  # opened read-only; never saved
        pdfbytes = rect_doc.convert_to_pdf()
        rect_doc.close()
        return fitz.open(stream=pdfbytes, filetype="pdf")
    return fitz.open(str(doc_path))


def process_file(
    source: str | Path,
    settings: Settings,
    *,
    out_dir: Optional[Path] = None,
    extra_detectors: Optional[List[DetectorFn]] = None,
    confirmed_boxes: Optional[Dict[int, List[Box]]] = None,
    ocr_verify: bool = False,
) -> FileResult:
    """Redact one file. `confirmed_boxes` (from the review UI) override detection.

    `extra_detectors` lets Phase 3 inject the GLiNER2 detector without this
    module importing it.
    """
    loaded = document_loader.load(source)
    audit = AuditReport(source_name=loaded.path.name, source_sha256=loaded.sha256, mode=settings.default_output_mode.value)

    detectors = _build_detectors(settings)
    if extra_detectors:
        detectors.extend(extra_detectors)

    doc = _open_readonly(loaded.path, loaded.doc_type)
    try:
        # Decide effective mode: Mode B requires a pure born-digital PDF.
        mode = settings.default_output_mode
        if mode == OutputMode.STRUCTURE_PRESERVING and not (
            loaded.doc_type == DocType.PDF
            and page_classifier.document_is_pure_born_digital(doc)
        ):
            mode = OutputMode.MAXIMUM_SAFE
        audit.mode = mode.value
        audit.pages = doc.page_count

        all_labels: List[str] = []
        forbidden_terms: List[str] = []
        page_results: List[PageResult] = []
        # Per-page boxes in their native space, kept separate per mode.
        pixel_pages: List[List[Box]] = []
        point_pages: Dict[int, List[Box]] = {}

        for i, page in enumerate(doc):
            kind = page_classifier.classify_page(page)
            page_text, space = _page_text_for(page, kind, settings)
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
                pixel_boxes = redaction_engine.to_pixel_boxes(boxes, space, settings.dpi)
                pixel_pages.append(pixel_boxes)
            else:
                point_pages[i] = boxes  # born-digital -> already PDF points

        # ----- Build output -----
        if mode == OutputMode.MAXIMUM_SAFE:
            images = []
            for i, page in enumerate(doc):
                img = pdf_renderer.render_page(page, dpi=settings.dpi)
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
            return FileResult(loaded.path, None, audit, success=False)

        out_path = redacted_output_path(loaded.path, out_dir)
        out_path.write_bytes(output_bytes)
        audit.output_name = out_path.name
        audit.output_sha256 = _sha256_bytes(output_bytes)

        if settings.save_audit_report:
            audit.save(out_path.with_suffix(".audit.json"))

        return FileResult(loaded.path, out_path, audit, success=True)
    finally:
        doc.close()
