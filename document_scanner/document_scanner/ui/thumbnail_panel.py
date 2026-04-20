from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem, QWidget

from document_scanner.models.document import DocumentImage


class ThumbnailPanel(QListWidget):
    order_changed = Signal(list)   # List[int] — new document indices in display order
    item_selected = Signal(int)    # int — index in the (current) documents list

    THUMBNAIL_SIZE = 150

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setIconSize(QSize(self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE))
        self.setSpacing(6)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setUniformItemSizes(True)
        self.setWordWrap(True)
        self.setFixedWidth(self.THUMBNAIL_SIZE + 40)

        self.currentRowChanged.connect(self._on_row_changed)

    # ── public API ────────────────────────────────────────────────────────────

    def add_document(self, doc: DocumentImage, index: int) -> None:
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, index)
        item.setText(doc.display_name)
        item.setToolTip(str(doc.path))
        src = doc.thumbnail if doc.thumbnail is not None else doc.original
        item.setIcon(self._make_icon(src))
        self.addItem(item)

    def update_thumbnail(self, index: int, processed: np.ndarray) -> None:
        for row in range(self.count()):
            if self.item(row).data(Qt.ItemDataRole.UserRole) == index:
                self.item(row).setIcon(self._make_icon(processed))
                return

    def clear_all(self) -> None:
        self.clear()

    # ── overrides ─────────────────────────────────────────────────────────────

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        new_order = [self.item(i).data(Qt.ItemDataRole.UserRole)
                     for i in range(self.count())]
        self.order_changed.emit(new_order)

    # ── private ───────────────────────────────────────────────────────────────

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            idx = self.item(row).data(Qt.ItemDataRole.UserRole)
            self.item_selected.emit(idx)

    def _make_icon(self, image: np.ndarray) -> QIcon:
        s = self.THUMBNAIL_SIZE
        thumb = cv2.resize(image, (s, s), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w * c, QImage.Format.Format_RGB888)
        return QIcon(QPixmap.fromImage(qimg.copy()))
