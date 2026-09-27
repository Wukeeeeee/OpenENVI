"""OpenENVI Resize Data & Spatial/Spectral Subsetting Dialog.

Provides interactive spatial crop coordinates, resolution resizing,
and spectral band selection.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.algorithms.subset import resize_subset_raster
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import RasterLayer, RasterMetadata


class ResizeWorker(QThread):
    """Background worker for executing spatial/spectral subset and resize."""

    progress = Signal(int, int)
    finished = Signal(str, np.ndarray, object)
    failed = Signal(str)

    def __init__(
        self,
        reader: BaseRasterReader,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        selected_bands: List[int],
        scale_factor: float,
        name: str,
    ):
        super().__init__()
        self.reader = reader
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.selected_bands = selected_bands
        self.scale_factor = scale_factor
        self.name = name

    def run(self):
        try:
            cube, meta = resize_subset_raster(
                reader=self.reader,
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
                selected_bands=self.selected_bands,
                scale_factor=self.scale_factor,
                progress_callback=lambda c, t: self.progress.emit(c, t),
            )
            self.finished.emit(self.name, cube, meta)
        except Exception as e:
            self.failed.emit(str(e))


class ResizeDataDialog(QDialog):
    """Dialog for resizing data and spatial/spectral subsetting."""

    result_generated = Signal(str, np.ndarray, object)

    def __init__(
        self,
        layer: RasterLayer,
        reader: BaseRasterReader,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader

        self.setWindowTitle(f"{tr('toolbox.tool_resize')} - {layer.name}")
        self.resize(550, 620)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        meta = self.layer.metadata
        w, h = meta.width, meta.height

        # 1. Spatial Subset Group
        grp_spatial = QGroupBox("Spatial Subset (Pixel Coordinates)")
        layout_spatial = QVBoxLayout(grp_spatial)

        grid_coords = QHBoxLayout()

        # Samples (X)
        v_x = QVBoxLayout()
        v_x.addWidget(QLabel("Samples (Width):"))
        h_x = QHBoxLayout()
        h_x.addWidget(QLabel("Start:"))
        self.sp_xmin = QSpinBox()
        self.sp_xmin.setRange(0, w - 1)
        self.sp_xmin.setValue(0)
        h_x.addWidget(self.sp_xmin)

        h_x.addWidget(QLabel("End:"))
        self.sp_xmax = QSpinBox()
        self.sp_xmax.setRange(1, w)
        self.sp_xmax.setValue(w)
        h_x.addWidget(self.sp_xmax)
        v_x.addLayout(h_x)
        grid_coords.addLayout(v_x)

        # Lines (Y)
        v_y = QVBoxLayout()
        v_y.addWidget(QLabel("Lines (Height):"))
        h_y = QHBoxLayout()
        h_y.addWidget(QLabel("Start:"))
        self.sp_ymin = QSpinBox()
        self.sp_ymin.setRange(0, h - 1)
        self.sp_ymin.setValue(0)
        h_y.addWidget(self.sp_ymin)

        h_y.addWidget(QLabel("End:"))
        self.sp_ymax = QSpinBox()
        self.sp_ymax.setRange(1, h)
        self.sp_ymax.setValue(h)
        h_y.addWidget(self.sp_ymax)
        v_y.addLayout(h_y)
        grid_coords.addLayout(v_y)

        layout_spatial.addLayout(grid_coords)

        # Reset button
        btn_reset = QPushButton("Reset to Full Scene")
        btn_reset.clicked.connect(self._reset_spatial)
        layout_spatial.addWidget(btn_reset)

        # Resample scale factor
        scale_box = QHBoxLayout()
        scale_box.addWidget(QLabel("Resize Scale Factor:"))
        self.cb_scale = QComboBox()
        self.cb_scale.addItem("1.0x (Original Resolution)", 1.0)
        self.cb_scale.addItem("0.5x (Downsample 2x)", 0.5)
        self.cb_scale.addItem("0.25x (Downsample 4x)", 0.25)
        self.cb_scale.addItem("2.0x (Upsample 2x)", 2.0)
        scale_box.addWidget(self.cb_scale)
        layout_spatial.addLayout(scale_box)

        layout.addWidget(grp_spatial)

        # 2. Spectral Subset Group
        grp_spectral = QGroupBox("Spectral Subset (Select Bands to Keep)")
        layout_spectral = QVBoxLayout(grp_spectral)

        self.list_bands = QListWidget()
        self.list_bands.setSelectionMode(QListWidget.MultiSelection)

        for b in range(meta.bands):
            b_name = (
                meta.band_details[b].name
                if meta.band_details and b < len(meta.band_details) and meta.band_details[b].name
                else f"Band {b + 1}"
            )
            item = QListWidgetItem(b_name)
            item.setData(Qt.UserRole, b)
            self.list_bands.addItem(item)
            item.setSelected(True)

        layout_spectral.addWidget(self.list_bands)

        btn_spectral_bar = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self.list_bands.selectAll)
        btn_spectral_bar.addWidget(btn_all)

        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self.list_bands.clearSelection)
        btn_spectral_bar.addWidget(btn_clear)
        layout_spectral.addLayout(btn_spectral_bar)

        layout.addWidget(grp_spectral)

        # 3. Output Name
        out_bar = QHBoxLayout()
        out_bar.addWidget(QLabel("Output Layer Name:"))
        self.txt_out_name = QLineEdit(f"{self.layer.name}_Subset")
        out_bar.addWidget(self.txt_out_name)
        layout.addLayout(out_bar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Action buttons
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton(tr("dialog.btn_ok"))
        self.btn_ok.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_ok.clicked.connect(self._start_resize)
        btn_bar.addWidget(self.btn_ok)

        layout.addLayout(btn_bar)

    def _reset_spatial(self):
        w = self.layer.metadata.width
        h = self.layer.metadata.height
        self.sp_xmin.setValue(0)
        self.sp_xmax.setValue(w)
        self.sp_ymin.setValue(0)
        self.sp_ymax.setValue(h)

    def _start_resize(self):
        selected_items = self.list_bands.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Bands Selected", "Please select at least one spectral band.")
            return

        selected_bands = [item.data(Qt.UserRole) for item in selected_items]
        selected_bands.sort()

        xmin = self.sp_xmin.value()
        xmax = self.sp_xmax.value()
        ymin = self.sp_ymin.value()
        ymax = self.sp_ymax.value()

        if xmin >= xmax or ymin >= ymax:
            QMessageBox.warning(self, "Invalid Bounds", "Start coordinate must be strictly less than End coordinate.")
            return

        scale_factor = float(self.cb_scale.currentData())
        name = self.txt_out_name.text().strip() or f"{self.layer.name}_Subset"

        self.btn_ok.setEnabled(False)
        self.progress_bar.setRange(0, len(selected_bands))
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.worker = ResizeWorker(
            reader=self.reader,
            x_min=xmin,
            x_max=xmax,
            y_min=ymin,
            y_max=ymax,
            selected_bands=selected_bands,
            scale_factor=scale_factor,
            name=name,
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, name: str, cube: np.ndarray, meta: object):
        self.result_generated.emit(name, cube, meta)
        self.accept()

    def _on_failed(self, err: str):
        self.btn_ok.setEnabled(True)
        self.progress_bar.hide()
        QMessageBox.critical(self, "Resize Failed", f"Failed to subset raster: {err}")
