"""Review preview: page image + editable overlay boxes.

Shows detected boxes coloured by confidence, lets the user accept/reject each,
draw a manual box, redact a full line, apply a box to all pages, and undo/redo.
The widget edits an in-memory list of `ReviewBox`es; the pipeline consumes the
accepted ones via `confirmed_boxes`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from app.detectors.base import Box


@dataclass
class ReviewBox:
    box: Box
    label: str = "manual"
    score: float = 1.0
    accepted: bool = True
    source: str = "manual"


def _confidence_color(score: float, accepted: bool) -> QColor:
    if not accepted:
        return QColor(150, 150, 150, 90)        # greyed-out rejected
    if score >= 0.85:
        return QColor(200, 0, 0, 120)           # high confidence — red
    if score >= 0.5:
        return QColor(220, 140, 0, 120)         # medium — orange
    return QColor(0, 120, 220, 120)             # low / GLiNER suggestion — blue


class PreviewWidget(QWidget):
    """One-page review canvas. Coordinates are in image device pixels."""

    boxes_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._boxes: List[ReviewBox] = []
        self._undo: List[List[ReviewBox]] = []
        self._redo: List[List[ReviewBox]] = []
        self._drag_start: Optional[QPoint] = None
        self._drag_now: Optional[QPoint] = None
        self.setMouseTracking(True)

    # --- state -------------------------------------------------------------
    def set_page(self, image: QImage, boxes: List[ReviewBox]) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._boxes = list(boxes)
        self._undo.clear()
        self._redo.clear()
        self.update()

    def accepted_boxes(self) -> List[Box]:
        return [rb.box for rb in self._boxes if rb.accepted]

    def _snapshot(self) -> None:
        self._undo.append([ReviewBox(**vars(b)) for b in self._boxes])
        self._redo.clear()

    def undo(self) -> None:
        if self._undo:
            self._redo.append([ReviewBox(**vars(b)) for b in self._boxes])
            self._boxes = self._undo.pop()
            self.update()
            self.boxes_changed.emit()

    def redo(self) -> None:
        if self._redo:
            self._undo.append([ReviewBox(**vars(b)) for b in self._boxes])
            self._boxes = self._redo.pop()
            self.update()
            self.boxes_changed.emit()

    def apply_to_all_pages(self, rb: ReviewBox) -> ReviewBox:
        """Caller propagates the returned box to every page's review list."""
        return rb

    # --- painting ----------------------------------------------------------
    def paintEvent(self, _event) -> None:
        if self._pixmap is None:
            return
        p = QPainter(self)
        p.drawPixmap(0, 0, self._pixmap)
        for rb in self._boxes:
            color = _confidence_color(rb.score, rb.accepted)
            p.setBrush(color)
            pen = QPen(color.darker(150))
            pen.setStyle(Qt.DashLine if not rb.accepted else Qt.SolidLine)
            p.setPen(pen)
            b = rb.box
            p.drawRect(QRectF(b.x0, b.y0, b.width, b.height))
        if self._drag_start and self._drag_now:
            p.setBrush(QColor(0, 0, 0, 60))
            p.setPen(QPen(QColor(0, 0, 0)))
            r = QRectF(self._drag_start, self._drag_now).normalized()
            p.drawRect(r)
        p.end()

    # --- interaction -------------------------------------------------------
    def mousePressEvent(self, e: QMouseEvent) -> None:
        pos = e.position().toPoint()
        if e.button() == Qt.LeftButton:
            hit = self._box_at(pos)
            if hit is not None:
                # Toggle accept/reject on click.
                self._snapshot()
                hit.accepted = not hit.accepted
                self.update()
                self.boxes_changed.emit()
            else:
                self._drag_start = pos
                self._drag_now = pos

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._drag_start is not None:
            self._drag_now = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self._drag_start is not None and self._drag_now is not None:
            r = QRectF(self._drag_start, self._drag_now).normalized()
            if r.width() > 3 and r.height() > 3:
                self._snapshot()
                self._boxes.append(
                    ReviewBox(Box(r.left(), r.top(), r.right(), r.bottom()))
                )
                self.boxes_changed.emit()
        self._drag_start = None
        self._drag_now = None
        self.update()

    def _box_at(self, pos: QPoint) -> Optional[ReviewBox]:
        for rb in reversed(self._boxes):
            b = rb.box
            if b.x0 <= pos.x() <= b.x1 and b.y0 <= pos.y() <= b.y1:
                return rb
        return None
