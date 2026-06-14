# LocalRedactor

A local-only, privacy-focused Windows desktop app that redacts words,
regex-matched identifiers, and AI-detected PII from PDFs and image scans, in
German and English, running entirely on CPU. **The original is never modified**,
and the redacted output (in the default mode) contains no extractable text —
copying from it yields nothing.

Everything runs offline. No network calls, no telemetry. Models are bundled or
cached locally and the HuggingFace stack is pinned to offline mode at startup.

---

## Hard Invariants

These override everything and are enforced in code:

1. **Inputs are never written, moved, or deleted.** Sources are opened read-only;
   outputs go to a separate `_redacted` file and never overwrite.
2. **Redacted output contains no recoverable sensitive data.** Default Mode A has
   no text layer at all — a black box that merely covers copyable text is a
   failure, not a redaction.
3. **Every export passes the verification gate** (text-extraction, raw-byte,
   metadata, optional OCR re-check) before it counts as done. On failure the
   output is discarded and the file is flagged for manual review.
4. **No network calls, ever** (unless the user opts into an update check).
   `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set before any ML import.
5. **No document text or detected PII is logged to disk.** Logs and audit
   reports hold settings, file names, counts, and status only.

---

## Quick start (development)

Windows:

```bat
run.bat
```

`run.bat` creates/reuses a `.venv`, installs `requirements.txt`, sets the offline
environment variables, and launches the Qt GUI.

Any platform (dev):

```bash
python -m pip install -r requirements.txt
python -m app.main
```

Run the tests:

```bash
python -m pytest
```

The test suite generates tiny born-digital and scanned PDF fixtures at runtime
(no binary blobs in the repo) and enforces the no-leak and original-untouched
invariants in CI.

---

## How it works

### Two output modes

- **Mode A — flatten to image (default, maximum safety).** Each page is rendered
  to a high-DPI image, redaction boxes are burned into the pixels, and pages are
  assembled into an image-only PDF with no text layer (`img2pdf`). Copying yields
  nothing by construction; works identically for scans.
- **Mode B — structure-preserving (optional, born-digital only).** PyMuPDF
  removes the matched characters from the content stream (not just covering
  them), keeping the rest of the document selectable. Automatically falls back to
  Mode A for any document that contains a scanned page.

Either way, document metadata (title, author, subject, keywords, producer, XMP,
embedded files, JavaScript) is stripped on export.

### Detection layers (merged, highest trust first)

1. **User words/phrases** — exact / case-insensitive / whole-word, with umlaut
   normalisation (`ä↔ae`, `ö↔oe`, `ü↔ue`, `ß↔ss`) and optional full-line redaction.
2. **Fuzzy OCR matching** (`rapidfuzz`) — tolerates OCR confusions so `Mu1ler`
   still matches `Müller`.
3. **German regex** — IBAN, tax IDs, social-security, eGK, BIC, phone, email,
   Kfz, etc. Broad patterns (dates, PLZ) are context-gated to avoid over-redaction.
4. **GLiNER2-PII** (`fastino/gliner2-pii-v1`, Apache 2.0) — a CPU-first semantic
   detector across 42 entity types and 7 languages incl. German. Long pages are
   processed with an overlapping sliding window and spans de-duplicated.
5. **Optional local LLM** (Phase 5) — suggestions only, never a final redaction.

All text detectors run against one reconstructed page string; `span_mapper`
translates the returned character spans into pixel/PDF boxes (the single piece of
real coordinate engineering, covered by `tests/test_ocr_boxes.py`).

### OCR engines

RapidOCR (ONNX, default, better on degraded scans) and Tesseract 5 (lightweight
baseline for clean German print) are both selectable and return word boxes.

---

## Project layout

See `app/` — `core/` (pipeline, rendering, OCR, span mapping, redaction,
verification, audit), `detectors/` (word, fuzzy, regex_de, gliner, llm),
`config/` (settings, paths, profiles), and `gui/` (PySide6 windows + background
worker). Settings and profiles persist under `%APPDATA%\LocalRedactor`.

---

## Licensing

The stack is almost fully permissive (Apache 2.0 / MIT / LGPL): RapidOCR,
GLiNER2, rapidfuzz, regex, pydantic, Pillow, img2pdf, and PySide6 (LGPL).

The **one copyleft dependency is PyMuPDF (AGPL-3.0)**, used for PDF rendering,
word boxes, and Mode B stream redaction. AGPL is fine for private/internal use.
If you distribute the source or offer it as a network service and do not want
AGPL obligations, either:

- obtain a commercial PyMuPDF license from Artifex, or
- swap the PDF backend (e.g. `pdfium`/`pypdfium2` for rendering plus another
  redaction path) and drop PyMuPDF.

This project's own metadata therefore declares `AGPL-3.0-only` to stay honest
about the combined work.

---

## Build phases

- **Phase 0** — skeleton, settings, paths, offline env, read-only loader. ✅
- **Phase 1** — Mode A redaction, OCR, span mapping, word detection, verification. ✅
- **Phase 2** — review preview, German regex, fuzzy matching, batch worker, audit,
  profiles. ✅ (detection + worker + review widget; wiring continues)
- **Phase 3** — GLiNER2-PII semantic layer with chunking + dedup. ✅
- **Phase 4** — hardening: encrypted/corrupt/mixed PDFs handled per file without
  aborting the batch, OCR-confidence warnings with auto re-run at higher DPI,
  secure temp wiping, status-only logging, PyInstaller build. ✅
- **Phase 5** — optional local LLM suggestions. ✅ (detector scaffold; opt-in)

## Packaging a standalone .exe

```bat
build.bat
```

Builds `dist\LocalRedactor\LocalRedactor.exe` via PyInstaller
(`packaging\LocalRedactor.spec`). A runtime hook pins the app offline before any
import, and `models\` is bundled for offline GLiNER2/OCR. Place the model assets
under `models\` before building.

## Robustness

Bad inputs never abort a batch: encrypted PDFs (try empty/supplied password),
corrupt files, and unreadable paths are each flagged with a status
(`encrypted` / `corrupt` / `error`) and the run continues. Scanned pages that
OCR poorly raise a confidence warning and are automatically re-OCR'd at a higher
DPI. All of this is recorded in the per-file audit report — counts and status
only, never document text.
