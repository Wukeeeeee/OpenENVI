"""OpenENVI Region of Interest (ROI) Tool Dialog.

Provides interactive ROI definition, polygon canvas drawing,
statistics calculation, mean spectral extraction, and mask raster layer generation.
"""

from typing import List, Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.i18n import tr
from core.models import RasterLayer
from core.roi import ROI


class ROIToolDialog(QDialog):
    """Dialog for defining ROIs, polygon drawing, statistics, and mask generation."""

    plot_mean_spectrum_requested = Signal(np.ndarray, Optional[np.ndarray], str, str)  # values, wavelengths, name, color
    mask_generated = Signal(str, np.ndarray, object)  # layer_name, 2d_mask_array, parent_metadata

    def __init__(self, layer: RasterLayer, reader, main_view=None, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self.main_view = main_view
        self.rois: List[ROI] = []
        self._pending_color = "#00ffcc"
        self._pending_idx = 1

        self.setWindowTitle(f"{tr('dialog.roi.title')} - {layer.name}")
        self.resize(720, 460)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Status & Drawing Instruction Label
        self.lbl_status = QLabel(
            "💡 提示：点击 [手绘多边形] 后在主视口左键添加顶点，双击或右键闭合多边形。"
        )
        self.lbl_status.setStyleSheet(
            "background-color: #2b2b2b; color: #a0c0e0; padding: 6px; border-radius: 4px; font-size: 12px;"
        )
        layout.addWidget(self.lbl_status)

        # ROI List Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            tr("layer_manager.col_name"), "Color",
            tr("dialog.roi.col_pixels"), tr("dialog.roi.col_mean"),
            tr("dialog.roi.col_min"), tr("dialog.roi.col_max")
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table, stretch=1)

        # Action Buttons
        btn_bar = QHBoxLayout()

        self.btn_draw_poly = QPushButton(f"✏️ {tr('dialog.roi.btn_draw_poly')}")
        self.btn_draw_poly.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 5px;")
        self.btn_draw_poly.clicked.connect(self._start_draw_polygon)
        btn_bar.addWidget(self.btn_draw_poly)

        self.btn_add = QPushButton(f"+ {tr('dialog.roi.btn_add_rect')}")
        self.btn_add.clicked.connect(self._add_roi)
        btn_bar.addWidget(self.btn_add)

        self.btn_delete = QPushButton("🗑️ 删除选中")
        self.btn_delete.clicked.connect(self._delete_roi)
        btn_bar.addWidget(self.btn_delete)

        self.btn_create_mask = QPushButton("🎭 生成掩膜图层")
        self.btn_create_mask.setStyleSheet("background-color: #d35400; color: white; font-weight: bold;")
        self.btn_create_mask.clicked.connect(self._create_mask_layer)
        btn_bar.addWidget(self.btn_create_mask)

        self.btn_stats = QPushButton(tr("dialog.roi.btn_stats"))
        self.btn_stats.clicked.connect(self._compute_stats)
        btn_bar.addWidget(self.btn_stats)

        self.btn_plot = QPushButton(tr("dialog.roi.btn_plot"))
        self.btn_plot.setStyleSheet("background-color: #3a71c1; color: white; font-weight: bold;")
        self.btn_plot.clicked.connect(self._plot_mean_spectrum)
        btn_bar.addWidget(self.btn_plot)

        self.btn_close = QPushButton(tr("dialog.btn_ok"))
        self.btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(self.btn_close)

        layout.addLayout(btn_bar)

        # Add default sample ROI covering center region
        w = layer.metadata.width
        h = layer.metadata.height
        default_roi = ROI(
            roi_id="roi_1",
            name="Sample ROI 1",
            color="#2ecc71",
            bbox=(w // 4, h // 4, 3 * w // 4, 3 * h // 4),
        )
        self.rois.append(default_roi)
        self._refresh_table()
        if self.main_view and hasattr(self.main_view, "redraw_rois"):
            self.main_view.redraw_rois(self.rois)

    def _refresh_table(self) -> None:
        """Update table items from self.rois list."""
        self.table.setRowCount(len(self.rois))
        for row, roi in enumerate(self.rois):
            item_name = QTableWidgetItem(roi.name)
            self.table.setItem(row, 0, item_name)

            item_color = QTableWidgetItem(roi.color)
            item_color.setBackground(QColor(roi.color))
            self.table.setItem(row, 1, item_color)

            # Placeholder stats
            for col in range(2, 6):
                if not self.table.item(row, col):
                    self.table.setItem(row, col, QTableWidgetItem("--"))

    def _add_roi(self) -> None:
        """Add a new rectangular ROI definition."""
        idx = len(self.rois) + 1
        colors = ["#e74c3c", "#3498db", "#9b59b6", "#f1c40f", "#e67e22"]
        color = colors[(idx - 1) % len(colors)]
        w = self.layer.metadata.width
        h = self.layer.metadata.height
        new_roi = ROI(
            roi_id=f"roi_{idx}",
            name=f"Sample ROI {idx}",
            color=color,
            bbox=(w // 3, h // 3, 2 * w // 3, 2 * h // 3),
        )
        self.rois.append(new_roi)
        self._refresh_table()
        if self.main_view and hasattr(self.main_view, "redraw_rois"):
            self.main_view.redraw_rois(self.rois)

    def _delete_roi(self) -> None:
        """Delete currently selected ROI."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self.rois):
            del self.rois[row]
            self._refresh_table()
            if self.main_view and hasattr(self.main_view, "redraw_rois"):
                self.main_view.redraw_rois(self.rois)
            self.lbl_status.setText(f"✓ 已删除 ROI #{row + 1}")

    def _create_mask_layer(self) -> None:
        """Generate a binary raster mask layer from selected ROI."""
        selected_rows = self.table.selectionModel().selectedRows()
        row = selected_rows[0].row() if selected_rows else 0
        if not (0 <= row < len(self.rois)):
            return

        roi = self.rois[row]
        w = self.layer.metadata.width
        h = self.layer.metadata.height
        mask_bool = roi.get_mask(h, w)
        mask_u8 = (mask_bool.astype(np.uint8) * 255)

        layer_name = f"{roi.name}_Mask"
        self.mask_generated.emit(layer_name, mask_u8, self.layer.metadata)
        self.lbl_status.setText(f"✓ 掩膜图层 [{layer_name}] 已成功生成并加入图层管理器！")

    def _compute_stats(self) -> None:
        """Calculate statistics across active band for all ROIs."""
        try:
            band_data = self.reader.read_band(0)
            for row, roi in enumerate(self.rois):
                stats = roi.calculate_statistics(band_data)
                self.table.setItem(row, 2, QTableWidgetItem(str(stats["count"])))
                self.table.setItem(row, 3, QTableWidgetItem(f"{stats['mean']:.4f}"))
                self.table.setItem(row, 4, QTableWidgetItem(f"{stats['min']:.4f}"))
                self.table.setItem(row, 5, QTableWidgetItem(f"{stats['max']:.4f}"))
            self.lbl_status.setText("✓ 已完成所有 ROI 像元统计指标计算。")
        except Exception as e:
            QMessageBox.critical(self, "Stats Error", f"Could not calculate statistics: {e}")

    def _plot_mean_spectrum(self) -> None:
        """Compute mean spectrum for selected ROI and emit signal."""
        selected_row = self.table.currentRow()
        if selected_row < 0 or selected_row >= len(self.rois):
            selected_row = 0

        if not (0 <= selected_row < len(self.rois)):
            return

        roi = self.rois[selected_row]
        try:
            mean_spec = roi.calculate_mean_spectrum(self.reader)
            if mean_spec is not None:
                wavelengths = np.array([
                    b.wavelength for b in self.layer.metadata.band_details
                    if b.wavelength is not None
                ], dtype=np.float32)
                if len(wavelengths) != len(mean_spec):
                    wavelengths = None

                self.plot_mean_spectrum_requested.emit(mean_spec, wavelengths, roi.name, roi.color)
                self.lbl_status.setText(f"✓ 已将 [{roi.name}] 均值波谱曲线投影到右下方波谱窗口。")
            else:
                QMessageBox.warning(self, "Empty ROI", "Selected ROI contains 0 pixels.")
        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"Failed to compute mean spectrum: {e}")

    def _start_draw_polygon(self) -> None:
        """Activate interactive canvas polygon drawing mode."""
        if not self.main_view:
            QMessageBox.information(self, "ROI Drawing", "Main canvas viewport is not connected.")
            return

        idx = len(self.rois) + 1
        colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f1c40f", "#e67e22", "#00ffcc"]
        color = colors[(idx - 1) % len(colors)]
        self._pending_color = color
        self._pending_idx = idx

        try:
            self.main_view.polygon_roi_completed.disconnect(self._on_polygon_completed)
        except Exception:
            pass
        self.main_view.polygon_roi_completed.connect(self._on_polygon_completed)

        self.main_view.start_drawing_polygon(color=color)
        self.lbl_status.setText(
            f"📍 [手绘中 - {color}] 在主视口点击左键添加顶点，双击或右键闭合多边形！"
        )

    def _on_polygon_completed(self, points: List[Tuple[float, float]]) -> None:
        """Receive closed polygon vertices and register ROI."""
        if len(points) < 3:
            return

        idx = getattr(self, "_pending_idx", len(self.rois) + 1)
        color = getattr(self, "_pending_color", "#2ecc71")
        new_roi = ROI(
            roi_id=f"polygon_roi_{idx}",
            name=f"Polygon ROI {idx}",
            color=color,
            polygon_points=points,
        )
        self.rois.append(new_roi)
        self._refresh_table()
        self._compute_stats()
        self.lbl_status.setText(
            f"✓ [Polygon ROI {idx}] 手绘闭合完成！像元统计已自动更新。"
        )
        self.raise_()
        self.activateWindow()
