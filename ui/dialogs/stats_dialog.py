"""OpenENVI Quick Statistics Dialog.

Displays multi-band summary statistics (Min, Max, Mean, Std Dev)
and interactive histogram distribution curves.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from core.algorithms.statistics import calculate_raster_statistics
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import RasterLayer


class StatsWorker(QThread):
    """Background worker for computing multi-band statistics."""

    progress = Signal(int, int)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, reader: BaseRasterReader):
        super().__init__()
        self.reader = reader

    def run(self):
        try:
            results = calculate_raster_statistics(
                self.reader,
                progress_callback=lambda c, t: self.progress.emit(c, t),
            )
            self.finished.emit(results)
        except Exception as e:
            self.failed.emit(str(e))


class QuickStatsDialog(QDialog):
    """Dialog displaying comprehensive statistics and histograms."""

    def __init__(
        self,
        layer: RasterLayer,
        reader: BaseRasterReader,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self._stats_results: List[Dict[str, Any]] = []

        self.setWindowTitle(f"{tr('toolbox.tool_stats')} - {layer.name}")
        self.resize(800, 520)

        self._init_ui()
        self._start_computation()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.layer.metadata.bands)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        splitter = QSplitter(Qt.Vertical)

        # 1. Statistics Table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            tr("dialog.roi.col_band"),
            tr("dialog.roi.col_min"),
            tr("dialog.roi.col_max"),
            tr("dialog.roi.col_mean"),
            tr("dialog.roi.col_stdev"),
            tr("dialog.roi.col_pixels"),
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        table_layout.addWidget(self.table)
        splitter.addWidget(table_container)

        # 2. Histogram Plot
        hist_container = QGroupBox(tr("display.histogram"))
        hist_layout = QVBoxLayout(hist_container)
        hist_layout.setContentsMargins(6, 6, 6, 6)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Pixel Value / DN")
        self.plot_widget.setLabel("left", "Frequency / Count")
        hist_layout.addWidget(self.plot_widget)

        splitter.addWidget(hist_container)
        splitter.setSizes([260, 240])
        layout.addWidget(splitter, stretch=1)

        # Button Bar
        btn_bar = QHBoxLayout()
        self.btn_export = QPushButton("Export Report (CSV)...")
        self.btn_export.clicked.connect(self._export_report)
        btn_bar.addWidget(self.btn_export)

        btn_bar.addStretch()

        self.btn_close = QPushButton(tr("dialog.btn_ok"))
        self.btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(self.btn_close)

        layout.addLayout(btn_bar)

    def _start_computation(self):
        self.worker = StatsWorker(self.reader)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_computation_finished)
        self.worker.failed.connect(self._on_computation_failed)
        self.worker.start()

    def _on_computation_finished(self, results: List[Dict[str, Any]]):
        self.progress_bar.hide()
        self._stats_results = results
        self.table.setRowCount(len(results))

        for row, s in enumerate(results):
            self.table.setItem(row, 0, QTableWidgetItem(s["band_name"]))
            self.table.setItem(row, 1, QTableWidgetItem(f"{s['min']:.4f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{s['max']:.4f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{s['mean']:.4f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{s['std']:.4f}"))
            self.table.setItem(row, 5, QTableWidgetItem(str(s["count"])))

        if results:
            self.table.selectRow(0)

    def _on_computation_failed(self, err_msg: str):
        self.progress_bar.hide()
        QMessageBox.critical(self, "Stats Error", f"Failed to compute statistics: {err_msg}")

    def _on_row_selected(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self._stats_results):
            self._plot_histogram(self._stats_results[row])

    def _plot_histogram(self, s: Dict[str, Any]):
        self.plot_widget.clear()
        counts = s["hist_counts"]
        edges = s["bin_edges"]
        if len(counts) == 0:
            return

        centers = (edges[:-1] + edges[1:]) / 2.0
        curve = pg.PlotCurveItem(
            centers,
            counts,
            pen=pg.mkPen(color="#3498db", width=2),
            fillLevel=0,
            brush=(52, 152, 219, 80),
        )
        self.plot_widget.addItem(curve)
        self.plot_widget.setTitle(f"Histogram: {s['band_name']}")

    def _export_report(self):
        if not self._stats_results:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Statistics Report",
            f"{self.layer.name}_statistics.csv",
            "CSV Files (*.csv);;Text Files (*.txt)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Band,Min,Max,Mean,StdDev,PixelCount\n")
                for s in self._stats_results:
                    f.write(f'"{s["band_name"]}",{s["min"]:.6f},{s["max"]:.6f},{s["mean"]:.6f},{s["std"]:.6f},{s["count"]}\n')
            QMessageBox.information(self, "Export Successful", f"Report saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not write file: {e}")
