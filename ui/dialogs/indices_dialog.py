"""OpenENVI Spectral Indices Dialog.

Allows users to calculate standard vegetation, water, and burn indices
(NDVI, NDWI, EVI, SAVI, NBR) with automatic wavelength-band matching.
"""

from typing import List, Optional
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.algorithms.indices import (
    calculate_evi,
    calculate_nbr,
    calculate_ndvi,
    calculate_ndwi,
    calculate_savi,
)
from core.i18n import tr
from core.models import RasterLayer


class IndicesDialog(QDialog):
    """Dialog for computing standard spectral remote sensing indices."""

    result_generated = Signal(str, np.ndarray, object)  # layer_name, 2D array, parent_meta

    def __init__(self, layer: RasterLayer, reader, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self.setWindowTitle(f"{tr('dialog.indices.title')} - {layer.name}")
        self.resize(500, 360)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Index Selection
        h_idx = QHBoxLayout()
        h_idx.addWidget(QLabel(tr("dialog.indices.select")))
        self.cb_index = QComboBox()
        self.cb_index.addItems([
            tr("indices.ndvi_name"),
            tr("indices.ndwi_name"),
            tr("indices.evi_name"),
            tr("indices.savi_name"),
            tr("indices.nbr_name"),
        ])
        self.cb_index.currentIndexChanged.connect(self._on_index_type_changed)
        h_idx.addWidget(self.cb_index, stretch=1)
        layout.addLayout(h_idx)

        # Band Assignment Form
        self.group_bands = QGroupBox(tr("dialog.indices.band_mapping"))
        self.form_bands = QFormLayout(self.group_bands)
        layout.addWidget(self.group_bands, stretch=1)

        self.cb_nir = QComboBox()
        self.cb_red = QComboBox()
        self.cb_green = QComboBox()
        self.cb_blue = QComboBox()
        self.cb_swir = QComboBox()

        band_labels = [b.display_name() for b in layer.metadata.band_details]
        if not band_labels:
            band_labels = [f"Band {i + 1}" for i in range(layer.metadata.bands)]

        for cb in [self.cb_nir, self.cb_red, self.cb_green, self.cb_blue, self.cb_swir]:
            cb.addItems(band_labels)

        # Auto-match bands based on wavelengths (nm / um) or names
        self._auto_select_bands()
        self._on_index_type_changed(0)

        # Action Buttons
        btn_box = QHBoxLayout()
        self.btn_compute = QPushButton(tr("dialog.indices.btn_compute"))
        self.btn_compute.setStyleSheet("background-color: #238636; color: white; font-weight: bold; padding: 6px;")
        self.btn_compute.clicked.connect(self._compute)
        btn_box.addWidget(self.btn_compute)

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        layout.addLayout(btn_box)

    def _auto_select_bands(self) -> None:
        """Find closest bands matching standard NIR, Red, Green, Blue, SWIR wavelengths or names."""
        bands = self.layer.metadata.band_details
        targets = {
            self.cb_blue: (480.0, ["blue", "coastal", "b2", "band 2"]),
            self.cb_green: (550.0, ["green", "b3", "band 3"]),
            self.cb_red: (660.0, ["red", "b4", "band 4"]),
            self.cb_nir: (840.0, ["nir", "near infrared", "b5", "band 5", "b8", "band 8"]),
            self.cb_swir: (2200.0, ["swir", "swir2", "b7", "band 7"]),
        }

        for cb, (target_wl, keywords) in targets.items():
            best_idx = None
            best_diff = float("inf")
            for b in bands:
                if b.wavelength is not None:
                    # Normalize wavelength to nanometers
                    wl_nm = b.wavelength
                    u = (b.wavelength_unit or "").lower()
                    if u in ("um", "µm", "micrometers", "micrometer") or wl_nm < 20.0:
                        wl_nm = wl_nm * 1000.0
                    diff = abs(wl_nm - target_wl)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = b.index

            if best_idx is not None and best_diff < 350.0:
                cb.setCurrentIndex(best_idx)
            else:
                # Fallback to name search
                for b in bands:
                    name_lower = (b.name or "").lower()
                    if any(kw in name_lower for kw in keywords):
                        cb.setCurrentIndex(b.index)
                        break

    def _on_index_type_changed(self, idx: int) -> None:
        """Update visible band selectors according to required inputs."""
        while self.form_bands.count():
            item = self.form_bands.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        if idx == 0:  # NDVI: NIR, Red
            self.form_bands.addRow(tr("dialog.indices.nir"), self.cb_nir)
            self.form_bands.addRow(tr("dialog.indices.red"), self.cb_red)
        elif idx == 1:  # NDWI: Green, NIR
            self.form_bands.addRow(tr("dialog.indices.green"), self.cb_green)
            self.form_bands.addRow(tr("dialog.indices.nir"), self.cb_nir)
        elif idx == 2:  # EVI: NIR, Red, Blue
            self.form_bands.addRow(tr("dialog.indices.nir"), self.cb_nir)
            self.form_bands.addRow(tr("dialog.indices.red"), self.cb_red)
            self.form_bands.addRow(tr("dialog.indices.blue"), self.cb_blue)
        elif idx == 3:  # SAVI: NIR, Red
            self.form_bands.addRow(tr("dialog.indices.nir"), self.cb_nir)
            self.form_bands.addRow(tr("dialog.indices.red"), self.cb_red)
        elif idx == 4:  # NBR: NIR, SWIR
            self.form_bands.addRow(tr("dialog.indices.nir"), self.cb_nir)
            self.form_bands.addRow(tr("dialog.indices.swir"), self.cb_swir)

    def _compute(self) -> None:
        """Calculate selected index and emit result layer."""
        idx = self.cb_index.currentIndex()
        try:
            nir = self.reader.read_band(self.cb_nir.currentIndex())
            red = self.reader.read_band(self.cb_red.currentIndex())

            if idx == 0:
                res = calculate_ndvi(nir, red)
                name = "NDVI"
            elif idx == 1:
                green = self.reader.read_band(self.cb_green.currentIndex())
                res = calculate_ndwi(green, nir)
                name = "NDWI"
            elif idx == 2:
                blue = self.reader.read_band(self.cb_blue.currentIndex())
                res = calculate_evi(nir, red, blue)
                name = "EVI"
            elif idx == 3:
                res = calculate_savi(nir, red)
                name = "SAVI"
            else:
                swir = self.reader.read_band(self.cb_swir.currentIndex())
                res = calculate_nbr(nir, swir)
                name = "NBR"

            self.result_generated.emit(f"Index: {name}", res, self.layer.metadata)
            self.accept()
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("indices.err_title"),
                f"{tr('indices.err_msg')}\n{e}",
            )
