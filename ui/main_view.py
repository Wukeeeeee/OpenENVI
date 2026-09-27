"""OpenENVI Main View Canvas.

High-performance raster display canvas built upon PyQtGraph and Qt Graphics View.
Provides pan, zoom, dynamic stretch enhancements, and a fully synchronized ENVI-style
Eagle-Eye overview inset with visible extent tracker.
"""

from typing import List, Optional, Tuple
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
    polygon_roi_completed = Signal(list)  # List[Tuple[float, float]]

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

        # ROI interactive drawing state
        self._is_drawing_roi: bool = False
        self._roi_draw_color: str = "#00ffcc"
        self._current_poly_pts: List[Tuple[float, float]] = []
        self._drawing_curve: Optional[pg.PlotDataItem] = None
        self._rubber_curve: Optional[pg.PlotCurveItem] = None
        self._roi_items: List[pg.GraphicsObject] = []

        # Connect scene mouse events
        self.glw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.image_item.mouseClickEvent = self._on_image_clicked
        self.view_box.sigRangeChanged.connect(self._on_main_view_range_changed)

        # Connect event bus
        event_bus.stretch_mode_changed.connect(self.set_stretch_mode)

        # Retranslate on creation and on language change
        self.retranslate_ui()
        from core.i18n import i18n
        i18n.language_changed.connect(lambda _: self.retranslate_ui())

    def retranslate_ui(self) -> None:
        """Update texts based on active language."""
        from core.i18n import tr
        self.lbl_overview.setText(tr("main_view.overview"))

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

        # Downsample for Eagle-Eye overview inset to maintain fast rendering on large rasters
        step = max(1, max(self._raster_h, self._raster_w) // 400)
        overview_data = stretched[::step, ::step] if step > 1 else stretched
        self.overview_img.setImage(
            overview_data,
            axisOrder="row-major",
            rect=QRectF(0, 0, self._raster_w, self._raster_h),
        )

        if reset_view:
            self.view_box.setRange(xRange=(0, self._raster_w), yRange=(0, self._raster_h), padding=0.02)
            self.overview_vb.setRange(xRange=(0, self._raster_w), yRange=(0, self._raster_h), padding=0.01)

        self.extent_roi.setVisible(True)
        self._update_extent_box()

    def clear(self) -> None:
        """Clear raster canvas, overlays, and overview window completely."""
        self.stop_drawing_polygon()
        self.clear_roi_overlays()
        self._raw_data = None
        self._raster_w = 0
        self._raster_h = 0
        self.image_item.clear()
        self.overview_img.clear()
        self.extent_roi.setVisible(False)

    def start_drawing_polygon(self, color: str = "#00ffcc") -> None:
        """Enter interactive polygon drawing mode."""
        self._is_drawing_roi = True
        self._roi_draw_color = color
        self._current_poly_pts.clear()

        if self._drawing_curve and self._drawing_curve in self.view_box.addedItems:
            self.view_box.removeItem(self._drawing_curve)
        if self._rubber_curve and self._rubber_curve in self.view_box.addedItems:
            self.view_box.removeItem(self._rubber_curve)

        self._drawing_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(color=color, width=2), symbol="o", symbolSize=6, symbolBrush=color
        )
        self.view_box.addItem(self._drawing_curve)

        self._rubber_curve = pg.PlotCurveItem([], [], pen=pg.mkPen(color=color, width=1.5, style=Qt.DashLine))
        self.view_box.addItem(self._rubber_curve)
        self.setCursor(Qt.CrossCursor)
        event_bus.status_message.emit("ROI Polygon: Left-click to add vertex, Right-click to close polygon", 5000)

    def stop_drawing_polygon(self) -> None:
        """Exit polygon drawing mode and clean up rubber band line."""
        self._is_drawing_roi = False
        self.setCursor(Qt.ArrowCursor)
        if self._drawing_curve and self._drawing_curve in self.view_box.addedItems:
            self.view_box.removeItem(self._drawing_curve)
            self._drawing_curve = None
        if self._rubber_curve and self._rubber_curve in self.view_box.addedItems:
            self.view_box.removeItem(self._rubber_curve)
            self._rubber_curve = None

    def add_roi_overlay(self, points: List[Tuple[float, float]], color: str = "#00ffcc") -> None:
        """Render a finished closed ROI polygon on the canvas."""
        if len(points) < 3:
            return
        xs = [p[0] for p in points] + [points[0][0]]
        ys = [p[1] for p in points] + [points[0][1]]
        item = pg.PlotDataItem(
            xs, ys, pen=pg.mkPen(color=color, width=2), symbol="o", symbolSize=5, symbolBrush=color
        )
        self.view_box.addItem(item)
        self._roi_items.append(item)

    def clear_roi_overlays(self) -> None:
        """Clear all drawn ROI overlays from canvas."""
        for item in self._roi_items:
            if item in self.view_box.addedItems:
                self.view_box.removeItem(item)
        self._roi_items.clear()

    def remove_roi_overlay(self, index: int) -> None:
        """Remove a single ROI item by index."""
        if 0 <= index < len(self._roi_items):
            item = self._roi_items.pop(index)
            if item in self.view_box.addedItems:
                self.view_box.removeItem(item)

    def redraw_rois(self, rois: list) -> None:
        """Clear and redraw all ROIs (polygons and bounding boxes)."""
        self.clear_roi_overlays()
        for roi in rois:
            if hasattr(roi, "polygon_points") and roi.polygon_points and len(roi.polygon_points) >= 3:
                self.add_roi_overlay(roi.polygon_points, color=roi.color)
            elif hasattr(roi, "bbox") and roi.bbox is not None:
                x0, y0, x1, y1 = roi.bbox
                box_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                self.add_roi_overlay(box_pts, color=roi.color)

    def save_view_image(self, output_path: str) -> bool:
        """Export the currently active stretched raster display to PNG/JPEG/BMP."""
        if self._raw_data is None:
            return False
        stretched = apply_stretch(self._raw_data, mode=self._stretch_mode)
        from PIL import Image
        pil_img = Image.fromarray(stretched)
        pil_img.save(output_path)
        return True

    def set_stretch_mode(self, mode: str) -> None:
        """Update contrast stretch mode and refresh display."""
        self._stretch_mode = mode
        if self._raw_data is not None:
            stretched = apply_stretch(self._raw_data, mode=self._stretch_mode)
            self.image_item.setImage(stretched, axisOrder="row-major")

            step = max(1, max(self._raster_h, self._raster_w) // 400)
            overview_data = stretched[::step, ::step] if step > 1 else stretched
            self.overview_img.setImage(
                overview_data,
                axisOrder="row-major",
                rect=QRectF(0, 0, self._raster_w, self._raster_h),
            )

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

        if self._is_drawing_roi and self._current_poly_pts and self._rubber_curve:
            last_x, last_y = self._current_poly_pts[-1]
            self._rubber_curve.setData([last_x, mouse_point.x()], [last_y, mouse_point.y()])

    def _on_image_clicked(self, event) -> None:
        """Handle mouse click on the image canvas."""
        pos = event.pos()
        fx = float(pos.x())
        fy = float(pos.y())

        if self._is_drawing_roi:
            if event.button() == Qt.LeftButton:
                if 0 <= fx < self._raster_w and 0 <= fy < self._raster_h:
                    self._current_poly_pts.append((fx, fy))
                    xs = [p[0] for p in self._current_poly_pts]
                    ys = [p[1] for p in self._current_poly_pts]
                    if self._drawing_curve:
                        self._drawing_curve.setData(xs, ys)
                    event.accept()
                    return
            elif event.button() == Qt.RightButton or event.double():
                if len(self._current_poly_pts) >= 3:
                    completed_pts = list(self._current_poly_pts)
                    color = self._roi_draw_color
                    self.stop_drawing_polygon()
                    self.add_roi_overlay(completed_pts, color=color)
                    self.polygon_roi_completed.emit(completed_pts)
                    event.accept()
                    return

        if event.button() == Qt.LeftButton:
            ix = int(fx)
            iy = int(fy)
            if 0 <= ix < self._raster_w and 0 <= iy < self._raster_h:
                event_bus.pixel_clicked.emit(ix, iy)
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
