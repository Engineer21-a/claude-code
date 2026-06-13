"""Background job runner.

Runs the redaction pipeline off the UI thread so the GUI never freezes. Files
are independent and OCR is the CPU bottleneck, so a process pool is used to fan
work across cores; progress is reported back through Qt signals.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from app.config.settings import Settings
from app.core.pipeline import FileResult, process_file


def _run_one(args):
    source, settings, out_dir = args
    # Runs in a worker process. Return a lightweight, picklable summary.
    result = process_file(source, settings, out_dir=out_dir)
    return {
        "source": str(result.source),
        "output": str(result.output) if result.output else None,
        "success": result.success,
        "passed": result.audit.verification_passed,
        "reasons": result.audit.verification_reasons,
        "boxes": result.audit.boxes_total,
    }


class BatchWorker(QThread):
    progress = Signal(int, int)        # done, total
    file_done = Signal(dict)           # per-file summary
    finished_all = Signal(list)        # list of summaries
    error = Signal(str)

    def __init__(
        self,
        files: List[Path],
        settings: Settings,
        out_dir: Optional[Path],
        max_workers: int = 0,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._files = list(files)
        self._settings = settings
        self._out_dir = out_dir
        self._max_workers = max_workers or None

    def run(self) -> None:  # noqa: D401 - QThread entrypoint
        results: List[dict] = []
        total = len(self._files)
        try:
            jobs = [(str(f), self._settings, self._out_dir) for f in self._files]
            with ProcessPoolExecutor(max_workers=self._max_workers) as ex:
                futures = [ex.submit(_run_one, j) for j in jobs]
                for i, fut in enumerate(as_completed(futures), start=1):
                    summary = fut.result()
                    results.append(summary)
                    self.file_done.emit(summary)
                    self.progress.emit(i, total)
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            self.error.emit(str(exc))
            return
        self.finished_all.emit(results)
