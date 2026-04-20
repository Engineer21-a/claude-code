from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)
from PySide6.QtGui import QPolygonF

from document_scanner.models.document import DocumentImage


def _ndarray_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg = QImage(rgb.data, w, h, w * c, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class CornerHandle(QGraphicsEllipseItem):
    """A draggable corner point overlaid on the document image."""

    RADIUS = 10
    COLOR_NORMAL = QColor(0, 150, 255, 200)
    COLOR_HOVER = QColor(255, 200, 0, 230)
    COLOR_BORDER = QColor(255, 255, 255, 200)

    def __init__(
        self,
        x: float,
        y: float,
        index: int,
        notify_fn: Callable[[], None],
    ) -> None:
        r = self.RADIUS
        super().__init__(-r, -r, r * 2, r * 2)
        self._index = index
        self._notify_fn = notify_fn

        self.setPos(x, y)
        self.setBrush(QBrush(self.COLOR_NORMAL))
        self.setPen(QPen(self.COLOR_BORDER, 2))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._notify_fn()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:
        self.setBrush(QBrush(self.COLOR_HOVER))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(QBrush(self.COLOR_NORMAL))
        super().hoverLeaveEvent(event)


class EditorView(QGraphicsView):
    """Center canvas: shows document image with 4 draggable corner handles."""

    corners_changed = Signal(list)   # List[QPointF]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._preview_pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._corner_handles: List[CornerHandle] = []
        self._quad_item: Optional[QGraphicsPolygonItem] = None

        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setRenderHint(self._scene.views()[0].renderHints() if self._scene.views() else 0)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))

    # ── public API ────────────────────────────────────────────────────────────

    def set_document(self, doc: DocumentImage) -> None:
        self._scene.clear()
        self._corner_handles = []
        self._quad_item = None
        self._preview_pixmap_item = None

        # Main image
        pixmap = _ndarray_to_pixmap(doc.original)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)

        # Corner handles
        for i, (x, y) in enumerate(doc.corners or []):
            handle = CornerHandle(float(x), float(y), i, self._on_corner_moved)
            self._scene.addItem(handle)
            self._corner_handles.append(handle)

        # Quad outline
        self._quad_item = QGraphicsPolygonItem()
        self._quad_item.setPen(QPen(QColor(0, 200, 255, 180), 2, Qt.PenStyle.DashLine))
        self._quad_item.setBrush(QBrush(QColor(0, 150, 255, 20)))
        self._quad_item.setZValue(5)
        self._scene.addItem(self._quad_item)
        self._update_quad_polygon()

        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_corners(self, corners: List[Tuple[int, int]]) -> None:
        for i, (x, y) in enumerate(corners):
            if i < len(self._corner_handles):
                self._corner_handles[i].setPos(float(x), float(y))
        self._update_quad_polygon()

    def get_corners(self) -> List[QPointF]:
        return [h.pos() for h in self._corner_handles]

    def update_preview(self, processed: np.ndarray) -> None:
        if self._pixmap_item is None:
            return
        pixmap = _ndarray_to_pixmap(processed)
        if self._preview_pixmap_item is None:
            self._preview_pixmap_item = QGraphicsPixmapItem(pixmap)
            self._preview_pixmap_item.setZValue(2)
            self._preview_pixmap_item.setOpacity(0.0)
            self._scene.addItem(self._preview_pixmap_item)
        else:
            self._preview_pixmap_item.setPixmap(pixmap)

    def show_preview(self, show: bool) -> None:
        if self._preview_pixmap_item:
            self._preview_pixmap_item.setOpacity(1.0 if show else 0.0)
        if self._pixmap_item:
            self._pixmap_item.setOpacity(0.0 if show else 1.0)
        for h in self._corner_handles:
            h.setVisible(not show)
        if self._quad_item:
            self._quad_item.setVisible(not show)

    # ── overrides ─────────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    # ── private ───────────────────────────────────────────────────────────────

    def _on_corner_moved(self) -> None:
        self._update_quad_polygon()
        self.corners_changed.emit(self.get_corners())

    def _update_quad_polygon(self) -> None:
        if not self._quad_item or not self._corner_handles:
            return
        poly = QPolygonF([h.pos() for h in self._corner_handles])
        self._quad_item.setPolygon(poly)
