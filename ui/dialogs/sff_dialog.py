"""OpenENVI Spectral Feature Fitting (SFF) Dialog.

Extracts endmember spectra from an image cube and, from the same basis,
least-squares abundance images -- the same pair of products ENVI's endmember
extraction returns.
"""

from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from core.algorithms.spectral import spectral_feature_fitting
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import RasterLayer, RasterMetadata

# Distinct colours so overlapping endmember curves stay readable.
ENDMEMBER_COLORS = ("#2ecc71", "#e74c3c", "#f1c40f", "#3498db", "#e67e22", "#9b59b6")


def least_squares_abundances(cube: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    """Unmix every pixel onto the endmember basis by ordinary least squares.

    Args:
        cube: 3D array of shape (bands, lines, samples).
        endmembers: 2D array of shape (num_endmembers, bands).

    Returns:
        3D array of shape (num_endmembers, lines, samples) of clamped abundances.
    """
    bands, lines, samples = cube.shape
    flat = cube.reshape(bands, -1).T
    basis = np.asarray(endmembers, dtype=np.float64).T          # (bands, k)
    abund = flat @ np.linalg.pinv(basis).T                     # (n, k)
    # A pixel of this scene is a non-negative mixture, so a negative coefficient is
    # the projection misfiring on noise rather than a real abundance.
    clamped = np.maximum(abund, 0.0)
    return clamped.T.reshape(basis.shape[1], lines, samples)


class SFFWorker(QThread):
    """Background worker that extracts endmembers and their abundance images."""

    progress = Signal(int, int)
    finished = Signal(str, np.ndarray, np.ndarray, object, object)
    failed = Signal(str)

    def __init__(
        self,
        reader: BaseRasterReader,
        name: str,
        num_features: int,
        max_samples: int,
        inlier_fraction: float,
        make_abundance: bool,
        seed: int = 42,
    ):
        super().__init__()
        self.reader = reader
        self.name = name
        self.num_features = num_features
        self.max_samples = max_samples
        self.inlier_fraction = inlier_fraction
        self.make_abundance = make_abundance
        self.seed = seed

    def run(self):
        try:
            meta = self.reader.metadata
            total_b = meta.bands
            h, w = meta.height, meta.width

            cube = np.empty((h, w, total_b), dtype=np.float32)
            for b in range(total_b):
                cube[:, :, b] = self.reader.read_band(b)
                self.progress.emit(b + 1, total_b)

            # spectral_feature_fitting takes (bands, lines, samples); MemoryRasterReader
            # and the rest of the app hand data over as (lines, samples, bands).
            ends, residuals = spectral_feature_fitting(
                cube.transpose(2, 0, 1),
                num_features=self.num_features,
                max_samples=self.max_samples,
                inlier_fraction=self.inlier_fraction,
                seed=self.seed,
            )

            abundance = None
            if self.make_abundance and len(ends) > 0:
                abundance = least_squares_abundances(cube.transpose(2, 0, 1), ends)

            abundance_meta = None
            if abundance is not None:
                abundance_meta = RasterMetadata(
                    width=w,
                    height=h,
                    bands=len(ends),
                    dtype="float32",
                    crs=meta.crs,
                    transform=meta.transform,
                    nodata=0.0,
                )
            self.finished.emit(self.name, ends, residuals, abundance, abundance_meta)
        except Exception as e:  # surfaced in the dialog rather than killing the thread
            self.failed.emit(str(e))


class SFFDialog(QDialog):
    """Dialog for launching Spectral Feature Fitting."""

    result_generated = Signal(str, np.ndarray, object)
    endmembers_extracted = Signal(str, np.ndarray, np.ndarray)

    def __init__(
        self,
        layer: RasterLayer,
        reader: BaseRasterReader,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self._worker: Optional[SFFWorker] = None

        self.setWindowTitle(f"{tr('toolbox.tool_sff')} - {layer.name}")
        self.resize(680, 560)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Preview of the extracted endmembers
        grp_preview = QGroupBox(tr("sff.grp_endmembers"))
        l_prev = QVBoxLayout(grp_preview)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("bottom", tr("spectral_profile.axis_x_wavelength"))
        self.plot_widget.setLabel("left", tr("spectral_profile.axis_y"))
        l_prev.addWidget(self.plot_widget)

        self.lbl_summary = QLabel(tr("sff.hint"))
        self.lbl_summary.setWordWrap(True)
        l_prev.addWidget(self.lbl_summary)

        layout.addWidget(grp_preview, stretch=1)

        # 2. Parameters
        grp_params = QGroupBox(tr("sff.grp_params"))
        form = QFormLayout(grp_params)

        self.spin_num = QSpinBox()
        self.spin_num.setRange(0, self.layer.metadata.bands)
        self.spin_num.setValue(0)
        self.spin_num.setSpecialValueText(tr("sff.auto"))
        self.spin_num.setToolTip(tr("sff.tip_num"))
        form.addRow(tr("sff.lbl_num"), self.spin_num)

        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(1000, 2000000)
        self.spin_samples.setSingleStep(5000)
        self.spin_samples.setValue(20000)
        self.spin_samples.setToolTip(tr("sff.tip_samples"))
        form.addRow(tr("sff.lbl_samples"), self.spin_samples)

        self.spin_inlier = QDoubleSpinBox()
        self.spin_inlier.setRange(0.01, 1.0)
        self.spin_inlier.setSingleStep(0.05)
        self.spin_inlier.setDecimals(2)
        self.spin_inlier.setValue(0.15)
        self.spin_inlier.setToolTip(tr("sff.tip_inlier"))
        form.addRow(tr("sff.lbl_inlier"), self.spin_inlier)

        self.chk_abundance = QCheckBox(tr("sff.chk_abundance"))
        self.chk_abundance.setChecked(True)
        self.chk_abundance.setToolTip(tr("sff.tip_abundance"))
        form.addRow("", self.chk_abundance)

        layout.addWidget(grp_params)

        # 3. Output name
        h_out = QHBoxLayout()
        h_out.addWidget(QLabel(tr("sff.out_name")))
        self.txt_out_name = QLineEdit(f"{self.layer.name}_SFF_Abundance")
        h_out.addWidget(self.txt_out_name)
        layout.addLayout(h_out)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # 4. Buttons
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton(tr("dialog.btn_ok"))
        self.btn_ok.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;"
        )
        self.btn_ok.clicked.connect(self._start)
        btn_bar.addWidget(self.btn_ok)

        layout.addLayout(btn_bar)

    def _axis(self) -> np.ndarray:
        """Wavelength axis for the plot, or band indices when unavailable."""
        details = self.layer.metadata.band_details
        if details and all(d.wavelength is not None for d in details):
            return np.array([d.wavelength for d in details], dtype=np.float64)
        return np.arange(1, self.layer.metadata.bands + 1, dtype=np.float64)

    def _start(self):
        meta = self.layer.metadata
        if meta.bands < 2:
            QMessageBox.warning(self, tr("dialog.error"), tr("sff.err_bands"))
            return

        name = self.txt_out_name.text().strip() or f"{self.layer.name}_SFF_Abundance"
        self.btn_ok.setEnabled(False)
        self.progress_bar.setRange(0, meta.bands)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self._worker = SFFWorker(
            reader=self.reader,
            name=name,
            num_features=self.spin_num.value(),
            max_samples=self.spin_samples.value(),
            inlier_fraction=self.spin_inlier.value(),
            make_abundance=self.chk_abundance.isChecked(),
        )
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_finished(self, name, ends, residuals, abundance, abundance_meta):
        self.progress_bar.hide()
        self.btn_ok.setEnabled(True)

        x = self._axis()
        for i, spectrum in enumerate(ends):
            color = ENDMEMBER_COLORS[i % len(ENDMEMBER_COLORS)]
            self.plot_widget.plot(
                x, spectrum, pen=pg.mkPen(color, width=2),
                name=f"{tr('sff.legend_em')} {i + 1}",
            )

        table = "\n".join(
            f"{tr('sff.legend_em')} {i + 1}: {r:.4f}" for i, r in enumerate(residuals)
        )
        self.lbl_summary.setText(f"{tr('sff.status_done')}\n{table}")

        self.endmembers_extracted.emit(name, ends, residuals)
        if abundance is not None and abundance_meta is not None:
            self.result_generated.emit(name, abundance.astype(np.float32), abundance_meta)
        self.accept()

    def _on_failed(self, err: str):
        self.progress_bar.hide()
        self.btn_ok.setEnabled(True)
        QMessageBox.critical(self, tr("dialog.error"), f"{tr('sff.err_failed')}: {err}")
