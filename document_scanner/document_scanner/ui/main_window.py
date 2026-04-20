from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image as PilImage
from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from document_scanner.models.document import DocumentImage, FilterSettings
from document_scanner.processing.dewarper import Dewarper
from document_scanner.processing.edge_detector import auto_detect_corners
from document_scanner.processing.enhancer import Enhancer
from document_scanner.processing.pipeline import ProcessingPipeline, PipelineWorker
from document_scanner.ui.editor_view import EditorView
from document_scanner.ui.filter_panel import FilterPanel
from document_scanner.ui.thumbnail_panel import ThumbnailPanel
from document_scanner.ui.toolbar import AppToolBar

_SUPPORTED_EXTENSIONS = (
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp", "*.heic",
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Document Scanner")
        self.setMinimumSize(1100, 700)

        self._documents: List[DocumentImage] = []
        self._current_index: int = -1
        self._pipeline = ProcessingPipeline(dewarper=Dewarper(), enhancer=Enhancer())
        self._worker: Optional[PipelineWorker] = None

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._update_preview)

        self._build_ui()
        self._connect_signals()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Toolbar
        self._toolbar = AppToolBar(self)
        self.addToolBar(self._toolbar)

        # Central splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        self._thumbnail_panel = ThumbnailPanel()
        splitter.addWidget(self._thumbnail_panel)

        self._editor_view = EditorView()
        splitter.addWidget(self._editor_view)

        self._filter_panel = FilterPanel()
        splitter.addWidget(self._filter_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("Ready")
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.setVisible(False)
        self._status_bar.addWidget(self._status_label)
        self._status_bar.addPermanentWidget(self._progress_bar)

    def _connect_signals(self) -> None:
        # Toolbar
        self._toolbar.load_images_requested.connect(self.load_images)
        self._toolbar.save_current_requested.connect(self.save_current)
        self._toolbar.process_all_requested.connect(self.process_all)
        self._toolbar.export_pdf_requested.connect(self.export_pdf)
        self._toolbar.export_images_requested.connect(self.export_images)
        self._toolbar.auto_detect_requested.connect(self._on_auto_detect_corners)
        self._toolbar.reset_corners_requested.connect(self._on_reset_corners)

        # Thumbnail panel
        self._thumbnail_panel.item_selected.connect(self._on_image_selected)
        self._thumbnail_panel.order_changed.connect(self._on_order_changed)

        # Editor view
        self._editor_view.corners_changed.connect(self._on_corners_changed)

        # Filter panel
        self._filter_panel.filter_changed.connect(self._on_filter_changed)
        self._filter_panel.dewarp_toggled.connect(self._on_dewarp_toggled)
        self._filter_panel.sharpen_toggled.connect(self._on_sharpen_toggled)
        self._filter_panel.apply_to_all_requested.connect(self._on_apply_to_all)

    # ── public slots ─────────────────────────────────────────────────────────

    def load_images(self, paths: Optional[List[Path]] = None) -> None:
        if paths is None:
            ext_filter = "Images (" + " ".join(_SUPPORTED_EXTENSIONS) + ")"
            files, _ = QFileDialog.getOpenFileNames(
                self, "Load Images", "", ext_filter
            )
            if not files:
                return
            paths = [Path(f) for f in files]

        for path in paths:
            try:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                doc = DocumentImage(path=path, original=img)
                self._documents.append(doc)
                self._thumbnail_panel.add_document(doc, len(self._documents) - 1)
            except Exception as exc:
                QMessageBox.warning(self, "Load Error", f"Could not load {path.name}:\n{exc}")

        if self._documents and self._current_index < 0:
            self._thumbnail_panel.setCurrentRow(0)

        self._toolbar.set_document_loaded(bool(self._documents))
        self._status_label.setText(f"{len(self._documents)} image(s) loaded")

    def save_current(self) -> None:
        if self._current_index < 0:
            return
        doc = self._documents[self._current_index]
        if doc.processed is None:
            doc.processed = self._pipeline.process(doc)
        default_name = doc.path.stem + "_scanned.png"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", default_name,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)",
        )
        if out_path:
            self.save_image_as(doc, Path(out_path))

    def save_image_as(self, doc: DocumentImage, path: Path) -> None:
        img = doc.processed if doc.processed is not None else doc.original
        cv2.imwrite(str(path), img)
        self._status_label.setText(f"Saved: {path.name}")

    def process_all(self) -> None:
        if not self._documents:
            return
        self._stop_worker()
        self._progress_bar.setMaximum(len(self._documents))
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._status_label.setText("Processing…")

        self._worker = PipelineWorker(self._pipeline, self._documents, preview=False)
        self._worker.progress.connect(self._on_process_progress)
        self._worker.finished.connect(self._on_process_all_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def export_pdf(self, path: Optional[Path] = None) -> None:
        if not self._documents:
            return
        self.process_all_sync()
        if path is None:
            out_path, _ = QFileDialog.getSaveFileName(
                self, "Export PDF", "scan.pdf", "PDF (*.pdf)"
            )
            if not out_path:
                return
            path = Path(out_path)

        pil_images = []
        for doc in self._documents:
            img = doc.processed if doc.processed is not None else doc.original
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(PilImage.fromarray(rgb))

        if not pil_images:
            return

        first, rest = pil_images[0], pil_images[1:]
        first.save(str(path), save_all=True, append_images=rest, resolution=200)
        self._status_label.setText(f"PDF exported: {path.name}")

    def export_images(self, directory: Optional[Path] = None) -> None:
        if not self._documents:
            return
        self.process_all_sync()
        if directory is None:
            dir_path = QFileDialog.getExistingDirectory(self, "Export Images To")
            if not dir_path:
                return
            directory = Path(dir_path)

        for doc in self._documents:
            img = doc.processed if doc.processed is not None else doc.original
            out = directory / (doc.path.stem + "_scanned.png")
            cv2.imwrite(str(out), img)
        self._status_label.setText(f"Images exported to {directory}")

    def process_all_sync(self) -> None:
        """Synchronous (blocking) process_all — used before export."""
        total = len(self._documents)
        for i, doc in enumerate(self._documents):
            if doc.needs_reprocess or doc.processed is None:
                self._pipeline.process(doc)
            self._status_label.setText(f"Processing {i + 1}/{total}…")

    # ── private slots ─────────────────────────────────────────────────────────

    def _on_image_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._documents):
            return
        self._current_index = index
        doc = self._documents[index]
        self._editor_view.set_document(doc)
        self._filter_panel.set_document(doc)
        if doc.processed is not None:
            self._editor_view.update_preview(doc.processed)
        self._status_label.setText(f"Editing: {doc.display_name}")

    def _on_order_changed(self, new_order: List[int]) -> None:
        self._documents = [self._documents[i] for i in new_order]
        # Update stored indices in thumbnail panel items
        for row in range(self._thumbnail_panel.count()):
            item = self._thumbnail_panel.item(row)
            item.setData(Qt.ItemDataRole.UserRole, row)
        # Sync current index
        if self._current_index >= 0:
            for row in range(self._thumbnail_panel.count()):
                if self._thumbnail_panel.currentRow() == row:
                    self._current_index = row

    def _on_corners_changed(self, corners: List[QPointF]) -> None:
        if self._current_index < 0:
            return
        doc = self._documents[self._current_index]
        doc.corners = [(int(round(p.x())), int(round(p.y()))) for p in corners]
        doc.needs_reprocess = True
        self._schedule_preview()

    def _on_filter_changed(self, settings: FilterSettings) -> None:
        if self._current_index < 0:
            return
        doc = self._documents[self._current_index]
        doc.filter_settings = settings
        doc.needs_reprocess = True
        self._schedule_preview()

    def _on_dewarp_toggled(self, enabled: bool) -> None:
        if self._current_index >= 0:
            self._documents[self._current_index].dewarp_enabled = enabled
            self._documents[self._current_index].needs_reprocess = True
            self._schedule_preview()

    def _on_sharpen_toggled(self, enabled: bool) -> None:
        if self._current_index >= 0:
            self._documents[self._current_index].sharpen_enabled = enabled
            self._documents[self._current_index].needs_reprocess = True
            self._schedule_preview()

    def _on_auto_detect_corners(self) -> None:
        if self._current_index < 0:
            return
        doc = self._documents[self._current_index]
        corners = auto_detect_corners(doc.original)
        doc.corners = corners
        self._editor_view.set_corners(corners)
        doc.needs_reprocess = True
        self._schedule_preview()

    def _on_reset_corners(self) -> None:
        if self._current_index < 0:
            return
        doc = self._documents[self._current_index]
        h, w = doc.original.shape[:2]
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        doc.corners = corners
        self._editor_view.set_corners(corners)
        doc.needs_reprocess = True
        self._schedule_preview()

    def _on_apply_to_all(self) -> None:
        if self._current_index < 0:
            return
        src = self._documents[self._current_index].filter_settings
        for doc in self._documents:
            doc.filter_settings = FilterSettings(**src.__dict__)
            doc.needs_reprocess = True
        self._status_label.setText("Filter settings applied to all documents")

    def _schedule_preview(self) -> None:
        self._preview_timer.start()

    def _update_preview(self) -> None:
        if self._current_index < 0:
            return
        doc = self._documents[self._current_index]
        self._stop_worker()

        self._worker = PipelineWorker(self._pipeline, [doc], preview=True)
        self._worker.finished.connect(self._on_preview_ready)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_preview_ready(self, result: np.ndarray) -> None:
        if result is None or self._current_index < 0:
            return
        self._editor_view.update_preview(result)
        self._thumbnail_panel.update_thumbnail(self._current_index, result)

    def _on_process_progress(self, current: int, total: int) -> None:
        self._progress_bar.setValue(current)
        self._status_label.setText(f"Processing {current}/{total}…")

    def _on_process_all_finished(self, _result) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText("Processing complete")
        # Refresh thumbnails
        for i, doc in enumerate(self._documents):
            if doc.processed is not None:
                self._thumbnail_panel.update_thumbnail(i, doc.processed)
        # Show preview for current doc
        if self._current_index >= 0:
            doc = self._documents[self._current_index]
            if doc.processed is not None:
                self._editor_view.update_preview(doc.processed)

    def _on_worker_error(self, message: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"Error: {message}")

    def _stop_worker(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(500)
