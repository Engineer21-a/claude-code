from __future__ import annotations

from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolBar, QWidget
from PySide6.QtCore import Signal


class AppToolBar(QToolBar):
    load_images_requested = Signal()
    save_current_requested = Signal()
    process_all_requested = Signal()
    export_pdf_requested = Signal()
    export_images_requested = Signal()
    auto_detect_requested = Signal()
    reset_corners_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Main Toolbar", parent)
        self.setMovable(False)

        self._act_load = self._add_action("Load Images", "document-open", self.load_images_requested)
        self.addSeparator()
        self._act_save = self._add_action("Save Current", "document-save", self.save_current_requested)
        self._act_export_pdf = self._add_action("Export PDF", "document-print", self.export_pdf_requested)
        self._act_export_img = self._add_action("Export Images", "image-x-generic", self.export_images_requested)
        self.addSeparator()
        self._act_auto_detect = self._add_action("Auto-Detect Edges", "edit-find", self.auto_detect_requested)
        self._act_reset = self._add_action("Reset Corners", "edit-undo", self.reset_corners_requested)
        self.addSeparator()
        self._act_process = self._add_action("Process All", "media-playback-start", self.process_all_requested)

        self.set_document_loaded(False)

    def set_document_loaded(self, loaded: bool) -> None:
        for act in (self._act_save, self._act_export_pdf, self._act_export_img,
                    self._act_auto_detect, self._act_reset, self._act_process):
            act.setEnabled(loaded)

    def _add_action(self, text: str, icon_name: str, signal: Signal) -> QAction:
        icon = QIcon.fromTheme(icon_name)
        act = QAction(icon, text, self)
        act.triggered.connect(signal)
        self.addAction(act)
        return act
