"""OpenENVI Main View Canvas.

High-performance raster display canvas built upon PyQtGraph and Qt Graphics View.
Provides pan, zoom, dynamic stretch enhancements, and a fully synchronized ENVI-style
Eagle-Eye overview inset with visible extent tracker.
"""

from typing import Optional
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from core.algorithms.stretch import apply_stretch
from core.events import event_bus
from core.i18n import tr


class MainViewWidget(QWidget):
    """Central raster display canvas with synchronized overview window."""

    view_resized = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MainViewWidget")

        # Root Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Primary pyqtgraph GraphicsLayoutWidget
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("#1e2124")
        layout.addWidget(self.glw)

        # Primary ViewBox and ImageItem
        self.view_box = self.glw.addViewBox(row=0, col=0)
        self.view_box.setAspectLocked(True)
        self.view_box.invertY(True)  # Standard image coordinates (0,0 top-left)

        self.image_item = pg.ImageItem()
        self.view_box.addItem(self.image_item)

        # Overview / Eagle-Eye Inset (Floating Frame inside canvas)
        self.overview_frame = QFrame(self)
        self.overview_frame.setObjectName("OverviewFrame")
        self.overview_frame.setStyleSheet(
            "QFrame#OverviewFrame { background-color: #282b30; border: 1px solid #4e535a; border-radius: 2px; }"
        )
        self.overview_frame.setFixedSize(200, 150)

        overview_layout = QVBoxLayout(self.overview_frame)
        overview_layout.setContentsMargins(2, 2, 2, 2)
        overview_layout.setSpacing(1)

        self.lbl_overview = QLabel("Overview", self.overview_frame)
        self.lbl_overview.setStyleSheet("font-size: 10px; color: #8e9297; font-weight: bold; padding: 2px;")
        overview_layout.addWidget(self.lbl_overview)

        self.overview_glw = pg.GraphicsLayoutWidget(self.overview_frame)
        self.overview_glw.setBackground("#1a1c1e")
        self.overview_vb = self.overview_glw.addViewBox(row=0, col=0, enableMenu=False)
        self.overview_vb.setAspectLocked(True)
        self.overview_vb.invertY(True)

        self.overview_img = pg.ImageItem()
        self.overview_vb.addItem(self.overview_img)

        # Overview Viewport Extent Box (Red Rectangle)
        self.extent_roi = pg.RectROI([0, 0], [10, 10], pen=pg.mkPen(color="#ff3333", width=1.5), movable=True)
        self.extent_roi.removeHandle(0)
        self.overview_vb.addItem(self.extent_roi)
        self.extent_roi.sigRegionChanged.connect(self._on_extent_roi_moved)

        overview_layout.addWidget(self.overview_glw)

        # Internal state
        self._raster_w = 0
        self._raster_h = 0
        self._raw_data: Optional[np.ndarray] = None
        self._stretch_mode = "Linear 2%"
        self._updating_roi = False

        # Connect scene mouse events
        self.glw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.image_item.mouseClickEvent = self._on_image_clicked
        self.view_box.sigRangeChanged.connect(self._on_main_view_range_changed)

        # Connect event bus
        event_bus.stretch_mode_changed.connect(self.set_stretch_mode)

    def resizeEvent(self, event) -> None:
        """Reposition overview inset to bottom-right corner."""
        super().resizeEvent(event)
        margin = 16
        fw = self.overview_frame.width()
        fh = self.overview_frame.height()
        self.overview_frame.move(self.width() - fw - margin, self.height() - fh - margin)

    def display_raster(
        self,
        raw_data: np.ndarray,
        reset_view: bool = True,
    ) -> None:
        """Display raw 2D (grayscale) or 3D (RGB) raster array with current stretch.

        Args:
            raw_data: np.ndarray of shape (H, W) or (H, W, 3).
            reset_view: Whether to auto-fit view to bounds.
        """
        self._raw_data = raw_data
        self._raster_h, self._raster_w = raw_data.shape[:2]

        stretched = apply_stretch(raw_data, mode=self._stretch_mode)

        self.image_item.setImage(stretched, axisOrder="row-major")
        self.overview_img.setImage(stretched, axisOrder="row-major")

        if reset_view:
            self.view_box.setRange(xRange=(0, self._raster_w), yRange=(0, self._raster_h), padding=0.02)
            self.overview_vb.setRange(xRange=(0, self._raster_w), yRange=(0, self._raster_h), padding=0.01)

        self.extent_roi.setVisible(True)
        self._update_extent_box()

    def clear(self) -> None:
        """Clear raster canvas and overview window completely."""
        self._raw_data = None
        self._raster_w = 0
        self._raster_h = 0
        self.image_item.clear()
        self.overview_img.clear()
        self.extent_roi.setVisible(False)

    def set_stretch_mode(self, mode: str) -> None:
        """Update contrast stretch mode and refresh display."""
        self._stretch_mode = mode
        if self._raw_data is not None:
            stretched = apply_stretch(self._raw_data, mode=self._stretch_mode)
            self.image_item.setImage(stretched, axisOrder="row-major")
            self.overview_img.setImage(stretched, axisOrder="row-major")

    def zoom_in(self) -> None:
        """Zoom in by 25%."""
        self.view_box.scaleBy((0.8, 0.8))

    def zoom_out(self) -> None:
        """Zoom out by 25%."""
        self.view_box.scaleBy((1.25, 1.25))

    def zoom_fit(self) -> None:
        """Fit image to current viewport."""
        if self._raster_w > 0 and self._raster_h > 0:
            self.view_box.setRange(xRange=(0, self._raster_w), yRange=(0, self._raster_h), padding=0.02)

    def _on_mouse_moved(self, pos: QPointF) -> None:
        """Track mouse coordinate over raster."""
        mouse_point = self.view_box.mapSceneToView(pos)
        x = int(mouse_point.x())
        y = int(mouse_point.y())

        if 0 <= x < self._raster_w and 0 <= y < self._raster_h:
            event_bus.pixel_hovered.emit(x, y)

    def _on_image_clicked(self, event) -> None:
        """Handle mouse click on the image canvas."""
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            x = int(pos.x())
            y = int(pos.y())
            if 0 <= x < self._raster_w and 0 <= y < self._raster_h:
                event_bus.pixel_clicked.emit(x, y)
        event.accept()

    def _on_main_view_range_changed(self) -> None:
        """Synchronize the overview extent box when main view pans or zooms."""
        self._update_extent_box()

    def _update_extent_box(self) -> None:
        """Update overview extent rectangle position and size."""
        if self._raster_w <= 0 or self._raster_h <= 0 or self._updating_roi:
            return

        self._updating_roi = True
        try:
            view_range = self.view_box.viewRange()
            x_min, x_max = view_range[0]
            y_min, y_max = view_range[1]

            rx = max(0, x_min)
            ry = max(0, y_min)
            rw = min(self._raster_w, x_max) - rx
            rh = min(self._raster_h, y_max) - ry

            if rw > 0 and rh > 0:
                self.extent_roi.setPos((rx, ry), update=False)
                self.extent_roi.setSize((rw, rh), update=False)
        finally:
            self._updating_roi = False

    def _on_extent_roi_moved(self) -> None:
        """Handle dragging of the overview extent rectangle to pan the main view."""
        if self._updating_roi or self._raster_w <= 0:
            return

        self._updating_roi = True
        try:
            pos = self.extent_roi.pos()
            size = self.extent_roi.size()
            rx, ry = pos.x(), pos.y()
            rw, rh = size.x(), size.y()

            self.view_box.setRange(xRange=(rx, rx + rw), yRange=(ry, ry + rh), padding=0.0)
        finally:
            self._updating_roi = False

    def retranslate_ui(self) -> None:
        """Update texts on language change."""
        self.lbl_overview.setText(tr("main_view.overview"))
