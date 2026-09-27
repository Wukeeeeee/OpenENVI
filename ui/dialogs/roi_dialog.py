"""OpenENVI Region of Interest (ROI) Tool Dialog.

Provides interactive ROI definition, polygon canvas drawing,
multi-band statistics calculation, mean spectral extraction,
mask raster layer generation, and persistent ROI set management.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QFileDialog,
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


class ROIMultiBandStatsDialog(QDialog):
    """Detailed multi-band statistics dialog for a specific ROI."""

    def __init__(self, roi: ROI, stats_or_reader: Any, parent=None):
        super().__init__(parent)
        self.roi = roi
        if isinstance(stats_or_reader, list):
            self.stats_list = stats_or_reader
        else:
            # Reader passed directly
            reader = stats_or_reader
            h = reader.metadata.height
            w = reader.metadata.width
            mask = roi.get_mask(h, w)
            self.stats_list = []
            for b in range(reader.metadata.bands):
                b_name = (
                    reader.metadata.band_details[b].display_name()
                    if reader.metadata.band_details and b < len(reader.metadata.band_details)
                    else f"Band {b + 1}"
                )
                band_data = reader.read_band(b)
                valid = band_data[mask]
                valid = valid[np.isfinite(valid)]
                if len(valid) > 0:
                    self.stats_list.append({
                        "band_name": b_name,
                        "min": float(np.min(valid)),
                        "max": float(np.max(valid)),
                        "mean": float(np.mean(valid)),
                        "std": float(np.std(valid)),
                        "count": len(valid),
                    })
                else:
                    self.stats_list.append({
                        "band_name": b_name,
                        "min": float("nan"),
                        "max": float("nan"),
                        "mean": float("nan"),
                        "std": float("nan"),
                        "count": 0,
                    })

        self.setWindowTitle(f"{tr('dialog.roi.stats_multiband_title')} - {roi.name}")
        self.resize(560, 400)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        count = self.stats_list[0]["count"] if self.stats_list else 0
        lbl = QLabel(tr("dialog.roi.stats_summary").format(name=roi.name, count=count))
        lbl.setStyleSheet("font-weight: bold; color: #58a6ff; font-size: 13px;")
        layout.addWidget(lbl)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            tr("dialog.roi.col_band"),
            tr("dialog.roi.col_min"),
            tr("dialog.roi.col_max"),
            tr("dialog.roi.col_mean"),
            tr("dialog.roi.col_stdev"),
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setRowCount(len(self.stats_list))
        for row, s in enumerate(self.stats_list):
            self.table.setItem(row, 0, QTableWidgetItem(s["band_name"]))
            self.table.setItem(row, 1, QTableWidgetItem(f"{s['min']:.4f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{s['max']:.4f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{s['mean']:.4f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{s['std']:.4f}"))
        layout.addWidget(self.table, stretch=1)

        btn_bar = QHBoxLayout()
        btn_export = QPushButton(tr("stats.btn_export"))
        btn_export.clicked.connect(self._export_csv)
        btn_bar.addWidget(btn_export)
        btn_bar.addStretch()

        btn_close = QPushButton(tr("dialog.btn_ok"))
        btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(btn_close)
        layout.addLayout(btn_bar)

    def _export_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            tr("stats.save_title"),
            f"{self.roi.name}_statistics.csv",
            "CSV Files (*.csv);;Text Files (*.txt)",
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Band,Min,Max,Mean,StdDev,PixelCount\n")
                for s in self.stats_list:
                    f.write(
                        f'"{s["band_name"]}",{s["min"]:.6f},{s["max"]:.6f},{s["mean"]:.6f},{s["std"]:.6f},{s["count"]}\n'
                    )
            QMessageBox.information(
                self,
                tr("dialog.export.success_title"),
                f"{tr('dialog.export.success_msg')}\n{file_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, tr("dialog.error"), f"{tr('stats.err_write')}: {e}")


class ROIToolDialog(QDialog):
    """Compact floating dialog for defining ROIs, multi-polygon drawing, statistics, and mask generation."""

    plot_mean_spectrum_requested = Signal(object, object, str, str)  # values, wavelengths, name, color
    mask_generated = Signal(str, object, object)  # layer_name, 2d_mask_array, parent_metadata
    roi_updated = Signal()  # Emitted when ROIs or polygons change

    def __init__(self, layer: RasterLayer, reader, main_view=None, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self.main_view = main_view
        self._is_updating_table = False

        # Bind directly to layer's persistent ROI list
        if not hasattr(self.layer, "rois") or self.layer.rois is None:
            self.layer.rois = []
        if not self.layer.rois:
            self.layer.rois.append(ROI(roi_id="roi_1", name="ROI #1", color="#2ecc71"))
        self.rois: List[ROI] = self.layer.rois

        self.setWindowTitle(f"{tr('dialog.roi.title')} - {layer.name}")
        self.resize(380, 380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Status & Drawing Instruction Label
        self.lbl_status = QLabel(tr("dialog.roi.tip_status"))
        self.lbl_status.setStyleSheet(
            "background-color: #2b2b2b; color: #a0c0e0; padding: 6px; border-radius: 4px; font-size: 11px;"
        )
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        # ROI List Table (compact 4 columns)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            tr("layer_manager.col_name"),
            tr("dialog.roi.col_color"),
            tr("dialog.roi.col_polys"),
            tr("dialog.roi.col_pixels"),
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        layout.addWidget(self.table, stretch=1)

        # Action Buttons Row 1 (Drawing & Category Creation)
        btn_bar1 = QHBoxLayout()
        btn_bar1.setSpacing(4)

        self.btn_draw_poly = QPushButton(tr("dialog.roi.btn_draw"))
        self.btn_draw_poly.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; padding: 5px 8px;"
        )
        self.btn_draw_poly.clicked.connect(self._toggle_draw_polygon)
        btn_bar1.addWidget(self.btn_draw_poly)

        self.btn_new_class = QPushButton(tr("dialog.roi.btn_new_class"))
        self.btn_new_class.setStyleSheet(
            "background-color: #2980b9; color: white; font-weight: bold; padding: 5px 8px;"
        )
        self.btn_new_class.clicked.connect(self._add_roi_class)
        btn_bar1.addWidget(self.btn_new_class)

        self.btn_undo_poly = QPushButton(tr("dialog.roi.btn_undo_poly"))
        self.btn_undo_poly.clicked.connect(self._undo_last_polygon)
        btn_bar1.addWidget(self.btn_undo_poly)

        self.btn_delete = QPushButton(tr("dialog.roi.btn_delete_class"))
        self.btn_delete.setStyleSheet("background-color: #7f1d1d; color: white; padding: 5px 8px;")
        self.btn_delete.clicked.connect(self._delete_roi)
        btn_bar1.addWidget(self.btn_delete)

        layout.addLayout(btn_bar1)

        # Action Buttons Row 2 (Color & Persistence)
        btn_bar2 = QHBoxLayout()
        btn_bar2.setSpacing(4)

        self.btn_change_color = QPushButton(tr("dialog.roi.btn_change_color"))
        self.btn_change_color.clicked.connect(self._pick_color_for_selected)
        btn_bar2.addWidget(self.btn_change_color)

        self.btn_save_rois = QPushButton(tr("dialog.roi.btn_save_rois"))
        self.btn_save_rois.clicked.connect(self._save_rois_to_file)
        btn_bar2.addWidget(self.btn_save_rois)

        self.btn_load_rois = QPushButton(tr("dialog.roi.btn_load_rois"))
        self.btn_load_rois.clicked.connect(self._load_rois_from_file)
        btn_bar2.addWidget(self.btn_load_rois)

        layout.addLayout(btn_bar2)

        # Action Buttons Row 3 (Multi-band Stats & Mask)
        btn_bar3 = QHBoxLayout()
        btn_bar3.setSpacing(4)

        self.btn_stats = QPushButton(tr("dialog.roi.btn_stats"))
        self.btn_stats.setStyleSheet("background-color: #1f538d; color: white; padding: 5px 6px;")
        self.btn_stats.clicked.connect(self._compute_multiband_stats)
        btn_bar3.addWidget(self.btn_stats)

        self.btn_plot = QPushButton(tr("dialog.roi.btn_plot"))
        self.btn_plot.setStyleSheet("background-color: #3a71c1; color: white; font-weight: bold; padding: 5px 6px;")
        self.btn_plot.clicked.connect(self._plot_mean_spectrum)
        btn_bar3.addWidget(self.btn_plot)

        self.btn_create_mask = QPushButton(tr("dialog.roi.btn_create_mask"))
        self.btn_create_mask.setStyleSheet("background-color: #d35400; color: white; padding: 5px 6px;")
        self.btn_create_mask.clicked.connect(self._create_mask_layer)
        btn_bar3.addWidget(self.btn_create_mask)

        layout.addLayout(btn_bar3)

        # Keyboard shortcuts
        QShortcut(QKeySequence.Delete, self, self._delete_roi)
        QShortcut(QKeySequence("Backspace"), self, self._undo_last_polygon)

        self._refresh_table()
        self.table.selectRow(0)

        # Connect polygon completed signal if main_view is available
        if self.main_view:
            self.main_view.polygon_roi_completed.connect(self._on_polygon_completed)
            if hasattr(self.main_view, "redraw_rois"):
                self.main_view.redraw_rois(self.rois)

    def _get_selected_row(self) -> int:
        """Get the currently selected or active row in table."""
        row = self.table.currentRow()
        if 0 <= row < len(self.rois):
            return row
        selected = self.table.selectedItems()
        if selected:
            return selected[0].row()
        return 0 if self.rois else -1

    def _refresh_table(self) -> None:
        """Update table items from self.rois list without triggering itemChanged loops."""
        self._is_updating_table = True
        self.table.setRowCount(len(self.rois))
        w = self.layer.metadata.width
        h = self.layer.metadata.height

        for row, roi in enumerate(self.rois):
            # Name (editable)
            item_name = QTableWidgetItem(roi.name)
            self.table.setItem(row, 0, item_name)

            # Color block
            item_color = QTableWidgetItem(roi.color)
            item_color.setFlags(item_color.flags() & ~Qt.ItemIsEditable)
            item_color.setBackground(QColor(roi.color))
            c = QColor(roi.color)
            item_color.setForeground(QColor("#000000" if c.lightness() > 128 else "#ffffff"))
            item_color.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, item_color)

            # Polygons count
            poly_count = roi.num_polygons
            item_polys = QTableWidgetItem(str(poly_count))
            item_polys.setFlags(item_polys.flags() & ~Qt.ItemIsEditable)
            item_polys.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, item_polys)

            # Pixel count
            if poly_count > 0 and w > 0 and h > 0:
                mask = roi.get_mask(h, w)
                pix_count = int(np.sum(mask))
            else:
                pix_count = 0
            item_pix = QTableWidgetItem(str(pix_count))
            item_pix.setFlags(item_pix.flags() & ~Qt.ItemIsEditable)
            item_pix.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 3, item_pix)

        self._is_updating_table = False

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        """Sync inline name edits back to the ROI dataclass."""
        if self._is_updating_table:
            return
        row = item.row()
        col = item.column()
        if col == 0 and 0 <= row < len(self.rois):
            new_name = item.text().strip()
            if new_name:
                self.rois[row].name = new_name
                self.roi_updated.emit()

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """Open color picker if clicking on color column."""
        if col == 1:
            self._pick_color_for_row(row)

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """Open color picker if double-clicking color cell."""
        if col == 1:
            self._pick_color_for_row(row)

    def _pick_color_for_selected(self) -> None:
        """Open color picker for currently selected row."""
        row = self._get_selected_row()
        if 0 <= row < len(self.rois):
            self._pick_color_for_row(row)

    def _pick_color_for_row(self, row: int) -> None:
        """Open color dialog and update ROI color."""
        if not (0 <= row < len(self.rois)):
            return
        current_color = QColor(self.rois[row].color)
        new_color = QColorDialog.getColor(current_color, self, tr("dialog.roi.color_picker_title"))
        if new_color.isValid():
            self.rois[row].color = new_color.name()
            self._refresh_table()
            self.table.selectRow(row)
            if self.main_view and hasattr(self.main_view, "redraw_rois"):
                self.main_view.redraw_rois(self.rois)
            if self.main_view and getattr(self.main_view, "_is_drawing_roi", False):
                self.main_view.start_drawing_polygon(color=self.rois[row].color)
            self.roi_updated.emit()

    def _add_roi_class(self) -> None:
        """Create and select a new ROI category class."""
        idx = len(self.rois) + 1
        colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#00bcd4"]
        color = colors[(idx - 1) % len(colors)]
        new_roi = ROI(
            roi_id=f"roi_{idx}",
            name=f"ROI #{idx}",
            color=color,
        )
        self.rois.append(new_roi)
        self._refresh_table()
        target_row = len(self.rois) - 1
        self.table.selectRow(target_row)
        self.lbl_status.setText(
            tr("dialog.roi.msg_class_created").format(name=new_roi.name)
        )

        if self.main_view and getattr(self.main_view, "_is_drawing_roi", False):
            self.main_view.start_drawing_polygon(color=new_roi.color)
        self.roi_updated.emit()

    def _delete_roi(self) -> None:
        """Delete currently selected ROI class and all its geometry."""
        row = self._get_selected_row()
        if 0 <= row < len(self.rois):
            deleted_name = self.rois[row].name
            del self.rois[row]
            self._refresh_table()
            if self.main_view and hasattr(self.main_view, "redraw_rois"):
                self.main_view.redraw_rois(self.rois)
            if self.rois:
                new_row = min(row, len(self.rois) - 1)
                self.table.selectRow(new_row)
            self.lbl_status.setText(tr("dialog.roi.msg_class_deleted").format(name=deleted_name))
            self.roi_updated.emit()

    def _undo_last_polygon(self) -> None:
        """Remove the most recently drawn polygon from selected ROI class."""
        row = self._get_selected_row()
        if not (0 <= row < len(self.rois)):
            return
        roi = self.rois[row]
        if roi.remove_last_polygon():
            self._refresh_table()
            if self.main_view and hasattr(self.main_view, "redraw_rois"):
                self.main_view.redraw_rois(self.rois)
            self.lbl_status.setText(
                tr("dialog.roi.msg_undo_poly").format(name=roi.name, count=roi.num_polygons)
            )
            self.roi_updated.emit()
        else:
            self.lbl_status.setText(tr("dialog.roi.msg_no_undo").format(name=roi.name))

    def _toggle_draw_polygon(self) -> None:
        """Toggle interactive polygon drawing on main canvas."""
        if not self.main_view:
            QMessageBox.information(self, tr("dialog.warning"), tr("dialog.roi.err_no_canvas"))
            return

        is_drawing = getattr(self.main_view, "_is_drawing_roi", False)
        if is_drawing:
            self.main_view.stop_drawing_polygon()
            self.btn_draw_poly.setText(tr("dialog.roi.btn_draw"))
            self.btn_draw_poly.setStyleSheet(
                "background-color: #27ae60; color: white; font-weight: bold; padding: 5px 8px;"
            )
            self.lbl_status.setText(tr("dialog.roi.msg_stopped"))
        else:
            row = self._get_selected_row()
            if row < 0 or row >= len(self.rois):
                self._add_roi_class()
                row = 0
            target_roi = self.rois[row]
            self.main_view.start_drawing_polygon(color=target_roi.color)
            self.btn_draw_poly.setText(tr("dialog.roi.btn_stop"))
            self.btn_draw_poly.setStyleSheet(
                "background-color: #c0392b; color: white; font-weight: bold; padding: 5px 8px;"
            )
            self.lbl_status.setText(
                tr("dialog.roi.msg_drawing").format(name=target_roi.name)
            )

    def _on_polygon_completed(self, points: List[Tuple[float, float]]) -> None:
        """Handle completed polygon closed in main view."""
        if len(points) < 3:
            return

        row = self._get_selected_row()
        if not (0 <= row < len(self.rois)):
            if not self.rois:
                self._add_roi_class()
            row = 0

        target_roi = self.rois[row]
        target_roi.add_polygon(points)
        self._refresh_table()
        self.table.selectRow(row)

        self.lbl_status.setText(
            tr("dialog.roi.msg_poly_added").format(
                name=target_roi.name, count=target_roi.num_polygons
            )
        )
        self.roi_updated.emit()

    def _compute_multiband_stats(self) -> None:
        """Calculate comprehensive statistics across all bands for the selected ROI."""
        row = self._get_selected_row()
        if not (0 <= row < len(self.rois)):
            QMessageBox.warning(self, tr("dialog.warning"), tr("dialog.roi.err_select_class"))
            return

        roi = self.rois[row]
        w = self.layer.metadata.width
        h = self.layer.metadata.height
        mask = roi.get_mask(h, w)
        num_pixels = int(np.sum(mask))

        if num_pixels == 0:
            QMessageBox.warning(
                self, tr("dialog.warning"), tr("dialog.roi.err_no_polygons").format(name=roi.name)
            )
            return

        total_bands = self.layer.metadata.bands
        stats_list = []

        try:
            for b in range(total_bands):
                b_name = (
                    self.layer.metadata.band_details[b].display_name()
                    if self.layer.metadata.band_details and b < len(self.layer.metadata.band_details)
                    else f"Band {b + 1}"
                )
                band_data = self.reader.read_band(b)
                valid = band_data[mask]
                valid = valid[np.isfinite(valid)]
                if len(valid) > 0:
                    stats_list.append({
                        "band_name": b_name,
                        "min": float(np.min(valid)),
                        "max": float(np.max(valid)),
                        "mean": float(np.mean(valid)),
                        "std": float(np.std(valid)),
                        "count": len(valid),
                    })
                else:
                    stats_list.append({
                        "band_name": b_name,
                        "min": float("nan"),
                        "max": float("nan"),
                        "mean": float("nan"),
                        "std": float("nan"),
                        "count": 0,
                    })

            self._stats_dlg = ROIMultiBandStatsDialog(roi, stats_list, parent=self)
            self._stats_dlg.show()
            self.lbl_status.setText(tr("dialog.roi.msg_stats_done").format(name=roi.name))
        except Exception as e:
            QMessageBox.critical(self, tr("dialog.error"), f"{tr('dialog.roi.err_stats_failed')} {e}")

    def _plot_mean_spectrum(self) -> None:
        """Compute mean spectrum for selected ROI across all bands and emit signal."""
        row = self._get_selected_row()
        if not (0 <= row < len(self.rois)):
            QMessageBox.warning(self, tr("dialog.warning"), tr("dialog.roi.err_select_class"))
            return

        roi = self.rois[row]
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
                self.lbl_status.setText(tr("dialog.roi.msg_spec_projected").format(name=roi.name))
            else:
                QMessageBox.warning(
                    self, tr("dialog.warning"), tr("dialog.roi.err_no_pixels").format(name=roi.name)
                )
        except Exception as e:
            QMessageBox.critical(self, tr("dialog.error"), f"{tr('dialog.roi.err_calc_spec_failed')} {e}")

    def _create_mask_layer(self) -> None:
        """Generate a binary raster mask layer unioning all polygons of selected ROI."""
        row = self._get_selected_row()
        if not (0 <= row < len(self.rois)):
            QMessageBox.warning(self, tr("dialog.warning"), tr("dialog.roi.err_select_class"))
            return

        roi = self.rois[row]
        w = self.layer.metadata.width
        h = self.layer.metadata.height
        mask_bool = roi.get_mask(h, w)
        num_pixels = int(np.sum(mask_bool))
        if num_pixels == 0:
            QMessageBox.warning(
                self, tr("dialog.warning"), tr("dialog.roi.err_no_polygons").format(name=roi.name)
            )
            return

        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        layer_name = f"{roi.name}_Mask"
        self.mask_generated.emit(layer_name, mask_u8, self.layer.metadata)
        self.lbl_status.setText(
            tr("dialog.roi.msg_mask_created").format(
                layer_name=layer_name, count=roi.num_polygons, pixels=num_pixels
            )
        )

    def _save_rois_to_file(self) -> None:
        """Export all ROIs of this layer to a JSON file."""
        if not self.rois:
            QMessageBox.warning(self, tr("dialog.warning"), tr("dialog.roi.err_no_rois_export"))
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            tr("dialog.roi.btn_save_rois"),
            f"{self.layer.name}_rois.json",
            "ROI Files (*.json);;All Files (*)",
        )
        if not file_path:
            return

        try:
            data = [roi.to_dict() for roi in self.rois]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(
                self,
                tr("dialog.export.success_title"),
                f"{tr('dialog.export.success_msg')}\n{file_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, tr("dialog.error"), f"{tr('dialog.roi.err_export_failed')} {e}")

    def _load_rois_from_file(self) -> None:
        """Import ROIs from a JSON file into current layer."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("dialog.roi.btn_load_rois"),
            "",
            "ROI Files (*.json);;All Files (*)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_rois = [ROI.from_dict(d) for d in data]
            if loaded_rois:
                self.rois.extend(loaded_rois)
                self._refresh_table()
                if self.main_view and hasattr(self.main_view, "redraw_rois"):
                    self.main_view.redraw_rois(self.rois)
                self.roi_updated.emit()
                self.lbl_status.setText(
                    tr("dialog.roi.msg_imported").format(count=len(loaded_rois))
                )
        except Exception as e:
            QMessageBox.critical(self, tr("dialog.error"), f"{tr('dialog.roi.err_import_failed')} {e}")

    def closeEvent(self, event):
        """When dialog is closed, stop drawing mode without destroying ROIs."""
        if self.main_view and getattr(self.main_view, "_is_drawing_roi", False):
            self.main_view.stop_drawing_polygon()
            self.btn_draw_poly.setText(tr("dialog.roi.btn_draw"))
        self.roi_updated.emit()
        event.accept()

