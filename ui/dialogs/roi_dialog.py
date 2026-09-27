"""OpenENVI Region of Interest (ROI) Tool Dialog.

Provides ROI definition, pixel statistics calculation, and mean spectral curve extraction.
"""

from typing import List, Optional
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.i18n import tr
from core.models import RasterLayer
from core.roi import ROI


class ROIToolDialog(QDialog):
    """Dialog for defining ROIs and computing multi-band statistics."""

    plot_mean_spectrum_requested = Signal(np.ndarray, Optional[np.ndarray], str, str)  # values, wavelengths, name, color

    def __init__(self, layer: RasterLayer, reader, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self.rois: List[ROI] = []
        self.setWindowTitle(tr("dialog.roi.title"))
        self.resize(600, 420)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

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

        self.btn_add = QPushButton(tr("dialog.roi.btn_add"))
        self.btn_add.clicked.connect(self._add_roi)
        btn_bar.addWidget(self.btn_add)

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
        """Add a new ROI definition."""
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

    def _compute_stats(self) -> None:
        """Calculate statistics across active band for all ROIs."""
        try:
            # Read first band for demonstration
            band_data = self.reader.read_band(0)
            for row, roi in enumerate(self.rois):
                stats = roi.calculate_statistics(band_data)
                self.table.setItem(row, 2, QTableWidgetItem(str(stats["count"])))
                self.table.setItem(row, 3, QTableWidgetItem(f"{stats['mean']:.4f}"))
                self.table.setItem(row, 4, QTableWidgetItem(f"{stats['min']:.4f}"))
                self.table.setItem(row, 5, QTableWidgetItem(f"{stats['max']:.4f}"))
        except Exception as e:
            QMessageBox.critical(self, "Stats Error", f"Could not calculate statistics: {e}")

    def _plot_mean_spectrum(self) -> None:
        """Compute mean spectrum for selected ROI and emit signal."""
        selected_row = self.table.currentRow()
        if selected_row < 0 or selected_row >= len(self.rois):
            selected_row = 0

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
                QMessageBox.information(
                    self,
                    "Mean Spectrum Plotted",
                    f"Mean spectral curve for '{roi.name}' was overlaid on the Spectral Profile dock!",
                )
            else:
                QMessageBox.warning(self, "Empty ROI", "Selected ROI contains 0 pixels.")
        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"Failed to compute mean spectrum: {e}")
