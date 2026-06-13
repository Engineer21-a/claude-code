"""Layer 5 (optional, Phase 5) — small local LLM, SUGGESTIONS ONLY.

A German-capable GGUF model via llama.cpp, temperature 0, JSON-only output, a
strict timeout and a thread cap. It answers only "what other categories on this
page might be sensitive" and every output is a weak signal a human confirms.

Hard rule: the LLM must NEVER place a final redaction by itself. This detector
returns suggestions tagged with a low score and is not wired into the automatic
box-burning path — the GUI surfaces them as unconfirmed candidates.
"""
from __future__ import annotations

import json
from typing import List, Optional

from app.detectors.base import Detection, PageText

_SYSTEM = (
    "Du bist ein Datenschutz-Assistent. Finde im Text mögliche sensible "
    "Kategorien. Antworte AUSSCHLIESSLICH mit JSON: eine Liste von Objekten "
    '{"text": "...", "category": "..."}. Keine Erklärungen.'
)


class LlmDetector:
    name = "llm"

    def __init__(
        self,
        model_path: str,
        n_threads: int = 4,
        timeout_s: float = 20.0,
        max_tokens: int = 512,
    ):
        self._model_path = model_path
        self._n_threads = n_threads
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._llm = None

    def _ensure(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("llama-cpp-python not installed.") from exc
        self._llm = Llama(
            model_path=self._model_path,
            n_threads=self._n_threads,
            n_ctx=2048,
            verbose=False,
        )

    def detect(self, page: PageText) -> List[Detection]:
        """Return weak SUGGESTION detections (low score, never auto-applied)."""
        self._ensure()
        prompt = f"{_SYSTEM}\n\nText:\n{page.text}\n\nJSON:"
        try:
            out = self._llm(  # type: ignore[misc]
                prompt, max_tokens=self._max_tokens, temperature=0.0, stop=["\n\n"]
            )
            raw = out["choices"][0]["text"]
            items = _parse_json(raw)
        except Exception:  # pragma: no cover - suggestions are best-effort
            return []

        detections: List[Detection] = []
        for item in items:
            text = item.get("text", "")
            if not text:
                continue
            idx = page.text.find(text)
            if idx < 0:
                continue
            detections.append(
                Detection(
                    start=idx,
                    end=idx + len(text),
                    label=f"llm:{item.get('category', 'unknown')}",
                    text=text,
                    score=0.3,  # weak signal — requires human confirmation
                    source=self.name,
                )
            )
        return detections


def _parse_json(raw: str) -> list:
    raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end < 0:
        return []
    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []
