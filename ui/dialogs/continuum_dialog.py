"""OpenENVI Continuum Removal Dialog.

Applies upper convex hull continuum removal across all image bands
or selected spectral subsets for quantitative mineralogy/absorption analysis.
"""

from typing import Optional
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from core.algorithms.continuum import continuum_removal_1d, continuum_removal_cube
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import RasterLayer, RasterMetadata


class ContinuumWorker(QThread):
    """Background worker for full cube continuum removal."""

    progress = Signal(int, int)
    finished = Signal(str, np.ndarray, object)
    failed = Signal(str)

    def __init__(self, reader: BaseRasterReader, name: str):
        super().__init__()
        self.reader = reader
        self.name = name

    def run(self):
        try:
            # Read full cube
            total_b = self.reader.metadata.bands
            h, w = self.reader.metadata.height, self.reader.metadata.width
            cube = np.empty((h, w, total_b), dtype=np.float32)
            for b in range(total_b):
                cube[:, :, b] = self.reader.read_band(b)

            wl = None
            if self.reader.metadata.band_details:
                w_list = [binfo.wavelength for binfo in self.reader.metadata.band_details if binfo.wavelength is not None]
                if len(w_list) == total_b:
                    wl = np.array(w_list, dtype=np.float32)

            cr_cube = continuum_removal_cube(
                cube,
                wavelengths=wl,
                progress_callback=lambda c, t: self.progress.emit(c, t),
            )

            meta = RasterMetadata(
                width=w,
                height=h,
                bands=total_b,
                dtype="float32",
                crs=self.reader.metadata.crs,
                transform=self.reader.metadata.transform,
                band_details=self.reader.metadata.band_details,
                raw_header=self.reader.metadata.raw_header,
            )
            self.finished.emit(self.name, cr_cube, meta)
        except Exception as e:
            self.failed.emit(str(e))


class ContinuumRemovalDialog(QDialog):
    """Dialog for launching Continuum Removal transform."""

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

        self.setWindowTitle(f"{tr('toolbox.tool_continuum')} - {layer.name}")
        self.resize(600, 480)

        self._init_ui()
        self._plot_preview()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Preview Graph Group
        grp_preview = QGroupBox("Spectral Sample Preview (Center Pixel)")
        l_prev = QVBoxLayout(grp_preview)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("bottom", "Band / Wavelength")
        self.plot_widget.setLabel("left", "Reflectance")
        l_prev.addWidget(self.plot_widget)

        layout.addWidget(grp_preview, stretch=1)

        # 2. Output name
        h_out = QHBoxLayout()
        h_out.addWidget(QLabel("Output Layer Name:"))
        self.txt_out_name = QLineEdit(f"{self.layer.name}_ContinuumRemoved")
        h_out.addWidget(self.txt_out_name)
        layout.addLayout(h_out)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # 3. Action Buttons
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton(tr("dialog.btn_ok"))
        self.btn_ok.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_ok.clicked.connect(self._start_continuum)
        btn_bar.addWidget(self.btn_ok)

        layout.addLayout(btn_bar)

    def _plot_preview(self):
        try:
            w, h = self.layer.metadata.width, self.layer.metadata.height
            spec = self.reader.read_pixel_profile(w // 2, h // 2)

            wl = None
            if self.layer.metadata.band_details:
                w_list = [b.wavelength for b in self.layer.metadata.band_details if b.wavelength is not None]
                if len(w_list) == len(spec):
                    wl = np.array(w_list, dtype=np.float32)

            x = wl if wl is not None else np.arange(1, len(spec) + 1)
            cr, hull = continuum_removal_1d(spec, wl)

            # Plot original
            self.plot_widget.plot(x, spec, pen=pg.mkPen("#e74c3c", width=2), name="Original Spectrum")
            # Plot hull
            self.plot_widget.plot(x, hull, pen=pg.mkPen("#f1c40f", width=1.5, style=Qt.DashLine), name="Convex Hull")
            # Plot CR
            self.plot_widget.plot(x, cr, pen=pg.mkPen("#2ecc71", width=2), name="Continuum Removed")
        except Exception:
            pass

    def _start_continuum(self):
        name = self.txt_out_name.text().strip() or f"{self.layer.name}_ContinuumRemoved"
        h = self.layer.metadata.height

        self.btn_ok.setEnabled(False)
        self.progress_bar.setRange(0, h)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.worker = ContinuumWorker(self.reader, name)
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
        QMessageBox.critical(self, "Processing Error", f"Continuum removal failed: {err}")
