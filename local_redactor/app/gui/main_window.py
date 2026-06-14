"""Main window: add files, set output folder, define words, run, open output.

Minimal but functional GUI satisfying the Phase 1 acceptance, with the review
preview (Phase 2) reachable from the queue. Heavy work runs on `BatchWorker`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import Settings, UserWord
from app.gui.worker import BatchWorker


class MainWindow(QMainWindow):
    def __init__(self, settings: Optional[Settings] = None):
        super().__init__()
        self.settings = settings or Settings.load()
        self._out_dir: Optional[Path] = None
        self._worker: Optional[BatchWorker] = None
        self.setWindowTitle("LocalRedactor")
        self.resize(820, 560)
        self._build_ui()

    # --- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        # File queue.
        root.addWidget(QLabel("Files to redact:"))
        self.file_list = QListWidget()
        root.addWidget(self.file_list, stretch=2)
        file_btns = QHBoxLayout()
        add_btn = QPushButton("Add files…")
        add_btn.clicked.connect(self._add_files)
        rm_btn = QPushButton("Remove selected")
        rm_btn.clicked.connect(self._remove_selected)
        file_btns.addWidget(add_btn)
        file_btns.addWidget(rm_btn)
        file_btns.addStretch()
        root.addLayout(file_btns)

        # Word list.
        root.addWidget(QLabel("Words/phrases to redact (one per line):"))
        self.words_edit = QListWidget()
        self.words_edit.setSelectionMode(QListWidget.ExtendedSelection)
        root.addWidget(self.words_edit, stretch=1)
        word_row = QHBoxLayout()
        self.word_input = QLineEdit()
        self.word_input.setPlaceholderText("e.g. Erika Mustermann")
        self.word_input.returnPressed.connect(self._add_word)
        add_word_btn = QPushButton("Add word")
        add_word_btn.clicked.connect(self._add_word)
        word_row.addWidget(self.word_input)
        word_row.addWidget(add_word_btn)
        root.addLayout(word_row)

        # Output folder.
        out_row = QHBoxLayout()
        self.out_label = QLabel("Output: alongside each source")
        out_btn = QPushButton("Choose output folder…")
        out_btn.clicked.connect(self._choose_out_dir)
        out_row.addWidget(self.out_label, stretch=1)
        out_row.addWidget(out_btn)
        root.addLayout(out_row)

        # Run + progress.
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Redact")
        self.run_btn.clicked.connect(self._run)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.progress, stretch=1)
        root.addLayout(run_row)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — all processing is local and offline.")

    # --- Actions -----------------------------------------------------------
    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add files", "", "Documents (*.pdf *.png *.jpg *.jpeg *.tif *.tiff)"
        )
        for p in paths:
            self.file_list.addItem(QListWidgetItem(p))

    def _remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def _add_word(self) -> None:
        text = self.word_input.text().strip()
        if text:
            self.words_edit.addItem(text)
            self.word_input.clear()

    def _choose_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self._out_dir = Path(d)
            self.out_label.setText(f"Output: {d}")

    def _collect_files(self) -> List[Path]:
        return [Path(self.file_list.item(i).text()) for i in range(self.file_list.count())]

    def _collect_words(self) -> List[UserWord]:
        return [
            UserWord(text=self.words_edit.item(i).text())
            for i in range(self.words_edit.count())
        ]

    def _run(self) -> None:
        files = self._collect_files()
        if not files:
            QMessageBox.warning(self, "No files", "Add at least one file to redact.")
            return

        run_settings = self.settings.model_copy(deep=True)
        run_settings.user_words = self._collect_words()

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)

        self._worker = BatchWorker(files, run_settings, self._out_dir)
        self._worker.progress.connect(lambda d, t: self.progress.setValue(d))
        self._worker.file_done.connect(self._on_file_done)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_file_done(self, summary: dict) -> None:
        name = Path(summary["source"]).name
        warn = f" ({len(summary.get('warnings') or [])} warning(s))" if summary.get("warnings") else ""
        if summary["success"]:
            self.statusBar().showMessage(
                f"{name}: redacted ({summary['boxes']} boxes), verified{warn}."
            )
        else:
            status = summary.get("status", "flagged")
            reason = summary.get("error") or "; ".join(
                summary.get("reasons") or ["verification failed"]
            )
            self.statusBar().showMessage(f"{name}: {status.upper()} — {reason}")

    def _on_finished(self, results: List[dict]) -> None:
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        ok = sum(1 for r in results if r["success"])
        failed = len(results) - ok
        warnings = sum(len(r.get("warnings") or []) for r in results)
        msg = f"Done. {ok} redacted, {failed} flagged for manual review."
        if warnings:
            msg += f" {warnings} warning(s) — see the audit reports."
        QMessageBox.information(self, "Finished", msg)
        self.statusBar().showMessage(msg)

    def _on_error(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", message)


def run_app(argv: Optional[List[str]] = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
