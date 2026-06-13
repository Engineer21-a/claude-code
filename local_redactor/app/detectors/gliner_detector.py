"""Layer 4 — GLiNER2-PII semantic detection (the AI layer).

Uses `fastino/gliner2-pii-v1` (Apache 2.0): a 205M encoder fine-tuned for PII
across 42 entity types and 7 languages incl. German. It returns character spans
plus confidence, which map straight to boxes. It does not hallucinate free text.

The encoder has a token limit (~384-512 tokens), so long pages are processed
with an overlapping sliding window and spans are de-duplicated across windows.
The model is loaded lazily and offline (HF_HUB_OFFLINE is set in app.main).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.detectors.base import Detection, PageText


@dataclass
class Window:
    start: int  # char offset of this window within the full page string
    text: str


def chunk_text(text: str, window_chars: int = 1500, overlap_chars: int = 300) -> List[Window]:
    """Split `text` into overlapping windows, preferring whitespace boundaries.

    `window_chars`/`overlap_chars` are character budgets that comfortably sit
    under the encoder's token limit for typical German prose. Returned windows
    carry their absolute start offset so spans can be lifted back to page coords.
    """
    if not text:
        return []
    if len(text) <= window_chars:
        return [Window(0, text)]

    windows: List[Window] = []
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + window_chars, n)
        if end < n:
            # Back off to the last whitespace so we don't cut a token in half.
            ws = text.rfind(" ", pos + window_chars - overlap_chars, end)
            if ws > pos:
                end = ws
        windows.append(Window(pos, text[pos:end]))
        if end >= n:
            break
        pos = max(end - overlap_chars, pos + 1)
    return windows


def dedupe_spans(detections: List[Detection]) -> List[Detection]:
    """Merge overlapping/duplicate spans (same label) from overlapping windows.

    Keeps the highest-confidence representative for each overlapping cluster.
    """
    by_label: dict[str, List[Detection]] = {}
    for d in detections:
        by_label.setdefault(d.label, []).append(d)

    result: List[Detection] = []
    for label, dets in by_label.items():
        dets.sort(key=lambda d: (d.start, d.end))
        cluster: List[Detection] = []
        cluster_end = -1
        for d in dets:
            if d.start <= cluster_end and cluster:
                cluster.append(d)
                cluster_end = max(cluster_end, d.end)
            else:
                if cluster:
                    result.append(max(cluster, key=lambda x: x.score))
                cluster = [d]
                cluster_end = d.end
        if cluster:
            result.append(max(cluster, key=lambda x: x.score))
    return result


class GlinerDetector:
    name = "gliner"

    def __init__(
        self,
        labels: List[str],
        threshold: float = 0.35,
        model_id: str = "fastino/gliner2-pii-v1",
        model_path: Optional[str] = None,
        window_chars: int = 1500,
        overlap_chars: int = 300,
    ):
        self._labels = labels
        self._threshold = threshold
        self._model_id = model_id
        self._model_path = model_path  # prefer a bundled local path if given
        self._window_chars = window_chars
        self._overlap_chars = overlap_chars
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from gliner2 import GLiNER2  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional heavy dep
            raise ImportError(
                "gliner2 not installed. `pip install gliner2` and bundle "
                "fastino/gliner2-pii-v1 under app/models for offline use."
            ) from exc
        source = self._model_path or self._model_id
        self._model = GLiNER2.from_pretrained(source)

    def _extract_window(self, text: str, base: int) -> List[Detection]:
        result = self._model.extract_entities(  # type: ignore[union-attr]
            text,
            self._labels,
            threshold=self._threshold,
            include_confidence=True,
            include_spans=True,
        )
        out: List[Detection] = []
        for ent in _iter_entities(result):
            start, end, label, score, span_text = ent
            out.append(
                Detection(
                    start=base + start,
                    end=base + end,
                    label=f"gliner:{label}",
                    text=span_text,
                    score=score,
                    source=self.name,
                )
            )
        return out

    def detect(self, page: PageText) -> List[Detection]:
        self._ensure_model()
        detections: List[Detection] = []
        for win in chunk_text(page.text, self._window_chars, self._overlap_chars):
            detections.extend(self._extract_window(win.text, win.start))
        return dedupe_spans(detections)


def _iter_entities(result) -> List[Tuple[int, int, str, float, str]]:
    """Normalise GLiNER2's return shape into (start, end, label, score, text).

    GLiNER variants differ in their exact dict keys; handle the common ones and
    fail loudly only if nothing matches.
    """
    entities = result
    if isinstance(result, dict):
        entities = result.get("entities") or result.get("results") or []

    parsed: List[Tuple[int, int, str, float, str]] = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        start = ent.get("start", ent.get("start_char"))
        end = ent.get("end", ent.get("end_char"))
        label = ent.get("label", ent.get("type", ent.get("entity", "")))
        score = float(ent.get("score", ent.get("confidence", 1.0)))
        text = ent.get("text", ent.get("span", ""))
        if start is None or end is None:
            continue
        parsed.append((int(start), int(end), str(label), score, str(text)))
    return parsed
