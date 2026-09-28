"""OpenENVI Pan-Sharpening Dialog.

Fuses lower-resolution multispectral bands with a high-resolution panchromatic band
using Gram-Schmidt or Brovey algorithms to synthesize a high-resolution color product.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

import rasterio
from core.algorithms.pansharpen import brovey_pansharpen, gram_schmidt_pansharpen
from core.events import event_bus
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterLayer, RasterMetadata


class PanSharpenWorker(QThread):
    """Background worker for executing Pan-Sharpening fusion."""

    progress = Signal(int, int)
    finished = Signal(str, np.ndarray, object)  # name, result_array, pan_metadata
    failed = Signal(str)

    def __init__(
        self,
        ms_reader: BaseRasterReader,
        selected_bands: List[int],
        pan_reader: Optional[BaseRasterReader],
        pan_file_path: Optional[str],
        method: str,
        name: str,
    ):
        super().__init__()
        self.ms_reader = ms_reader
        self.selected_bands = selected_bands
        self.pan_reader = pan_reader
        self.pan_file_path = pan_file_path
        self.method = method
        self.name = name

    def run(self):
        try:
            # 1. Read Pan band
            pan_meta = None
            if self.pan_reader is not None:
                pan_band = self.pan_reader.read_band(0)
                pan_meta = self.pan_reader.metadata
            elif self.pan_file_path and os.path.isfile(self.pan_file_path):
                with rasterio.open(self.pan_file_path) as ds:
                    pan_band = ds.read(1).astype(np.float32)
                    pan_meta = RasterMetadata(
                        width=ds.width,
                        height=ds.height,
                        bands=1,
                        dtype="float32",
                        crs=ds.crs.to_string() if ds.crs else None,
                        transform=tuple(ds.transform) if ds.transform else None,
                    )
            else:
                raise ValueError("No valid Panchromatic source specified.")

            # 2. Read selected MS bands
            ms_bands = []
            for b in self.selected_bands:
                ms_bands.append(self.ms_reader.read_band(b))

            # 3. Execute Pan-Sharpening algorithm
            if self.method == "brovey":
                fused = brovey_pansharpen(
                    pan_band=pan_band,
                    ms_bands=ms_bands,
                    progress_callback=lambda s, t: self.progress.emit(s, t),
                )
            else:
                fused = gram_schmidt_pansharpen(
                    pan_band=pan_band,
                    ms_bands=ms_bands,
                    progress_callback=lambda s, t: self.progress.emit(s, t),
                )

            # 4. Construct enriched metadata preserving Pan spatial grid and MS spectral details
            fused_band_details = []
            ms_meta = self.ms_reader.metadata
            for idx, b in enumerate(self.selected_bands):
                if ms_meta and ms_meta.band_details and b < len(ms_meta.band_details):
                    orig_b = ms_meta.band_details[b]
                    fused_band_details.append(
                        BandInfo(
                            index=idx,
                            name=orig_b.name or f"Fused Band {b + 1}",
                            wavelength=orig_b.wavelength,
                            wavelength_unit=orig_b.wavelength_unit,
                            fwhm=orig_b.fwhm,
                        )
                    )
                else:
                    fused_band_details.append(BandInfo(index=idx, name=f"Fused Band {b + 1}"))

            fused_meta = RasterMetadata(
                width=pan_meta.width if pan_meta else fused.shape[1],
                height=pan_meta.height if pan_meta else fused.shape[0],
                bands=len(self.selected_bands),
                dtype=str(fused.dtype),
                crs=pan_meta.crs if pan_meta else (ms_meta.crs if ms_meta else None),
                transform=pan_meta.transform if pan_meta else (ms_meta.transform if ms_meta else None),
                nodata=ms_meta.nodata if ms_meta else None,
                band_details=fused_band_details,
                default_bands=(0, 1, 2) if len(self.selected_bands) >= 3 else (0,),
                raw_header=dict(ms_meta.raw_header) if ms_meta and ms_meta.raw_header else {},
            )

            self.finished.emit(self.name, fused, fused_meta)
        except Exception as e:
            self.failed.emit(str(e))


class PanSharpenDialog(QDialog):
    """Dialog for configuring and running Pan-Sharpening image fusion."""

    result_generated = Signal(str, np.ndarray, object)

    def __init__(
        self,
        available_layers: Dict[str, Tuple[RasterLayer, BaseRasterReader]],
        active_layer_id: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.available_layers = available_layers
        self.active_layer_id = active_layer_id or (next(iter(available_layers)) if available_layers else None)
        self._worker: Optional[PanSharpenWorker] = None

        self.setWindowTitle(tr("dialog.pansharpen.title"))
        self.resize(540, 600)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # 1. Low-Resolution Multispectral Source
        grp_ms = QGroupBox(tr("dialog.pansharpen.grp_ms"))
        lay_ms = QVBoxLayout(grp_ms)

        row_ms = QHBoxLayout()
        row_ms.addWidget(QLabel(tr("dialog.pansharpen.ms_layer")))
        self.cb_ms_layers = QComboBox()
        for lid, (lyr, _) in self.available_layers.items():
            self.cb_ms_layers.addItem(lyr.name, lid)
            if lid == self.active_layer_id:
                self.cb_ms_layers.setCurrentIndex(self.cb_ms_layers.count() - 1)
        self.cb_ms_layers.currentIndexChanged.connect(self._on_ms_layer_changed)
        row_ms.addWidget(self.cb_ms_layers)
        lay_ms.addLayout(row_ms)

        lay_ms.addWidget(QLabel(tr("dialog.pansharpen.select_ms_bands")))
        self.ms_band_list = QListWidget()
        lay_ms.addWidget(self.ms_band_list)
        main_layout.addWidget(grp_ms)

        # 2. High-Resolution Panchromatic Source
        grp_pan = QGroupBox(tr("dialog.pansharpen.grp_pan"))
        lay_pan = QVBoxLayout(grp_pan)

        self.bg_pan = QButtonGroup(self)
        self.rb_pan_layer = QRadioButton(tr("dialog.pansharpen.pan_from_layer"))
        self.rb_pan_file = QRadioButton(tr("dialog.pansharpen.pan_from_file"))
        self.bg_pan.addButton(self.rb_pan_layer)
        self.bg_pan.addButton(self.rb_pan_file)

        # Pan layer picker
        row_pan_layer = QHBoxLayout()
        row_pan_layer.addWidget(self.rb_pan_layer)
        self.cb_pan_layers = QComboBox()
        for lid, (lyr, _) in self.available_layers.items():
            self.cb_pan_layers.addItem(lyr.name, lid)
        row_pan_layer.addWidget(self.cb_pan_layers)
        lay_pan.addLayout(row_pan_layer)

        # Pan file picker
        row_pan_file = QHBoxLayout()
        row_pan_file.addWidget(self.rb_pan_file)
        self.txt_pan_file = QLineEdit()
        self.txt_pan_file.setPlaceholderText(tr("dialog.pansharpen.pan_placeholder"))
        self.btn_browse_pan = QPushButton(tr("dialog.export.btn_browse"))
        self.btn_browse_pan.clicked.connect(self._browse_pan_file)
        row_pan_file.addWidget(self.txt_pan_file)
        row_pan_file.addWidget(self.btn_browse_pan)
        lay_pan.addLayout(row_pan_file)

        self.rb_pan_layer.toggled.connect(self._on_pan_source_toggled)
        main_layout.addWidget(grp_pan)

        # 3. Fusion Method & Output
        grp_method = QGroupBox(tr("dialog.pansharpen.grp_method"))
        lay_method = QVBoxLayout(grp_method)

        row_algo = QHBoxLayout()
        row_algo.addWidget(QLabel(tr("dialog.pansharpen.algorithm")))
        self.cb_method = QComboBox()
        self.cb_method.addItem("Gram-Schmidt (ENVI Standard)", "gs")
        self.cb_method.addItem("Brovey Transform", "brovey")
        row_algo.addWidget(self.cb_method)
        lay_method.addLayout(row_algo)

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel(tr("dialog.pansharpen.out_name")))
        self.txt_name = QLineEdit()
        self.txt_name.setText("PanSharpened_Image")
        row_name.addWidget(self.txt_name)
        lay_method.addLayout(row_name)

        main_layout.addWidget(grp_method)

        # 4. Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 5. Buttons
        btn_bar = QHBoxLayout()
        self.btn_fuse = QPushButton(tr("dialog.pansharpen.btn_fuse"))
        self.btn_fuse.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_fuse.clicked.connect(self._start_fusion)
        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addStretch()
        btn_bar.addWidget(self.btn_fuse)
        btn_bar.addWidget(self.btn_cancel)
        main_layout.addLayout(btn_bar)

        # Initialize state
        self._populate_ms_bands()
        self._auto_detect_panchromatic()

    def _on_ms_layer_changed(self) -> None:
        self._populate_ms_bands()
        self._auto_detect_panchromatic()

    def _populate_ms_bands(self) -> None:
        """Populate MS band list, defaulting to true-color RGB or first 3 bands."""
        self.ms_band_list.clear()
        lid = self.cb_ms_layers.currentData()
        if not lid or lid not in self.available_layers:
            return

        lyr, rdr = self.available_layers[lid]
        meta = rdr.metadata
        default_bands = meta.default_bands if (meta.default_bands and len(meta.default_bands) == 3) else (0, 1, 2)

        for b in range(meta.bands):
            b_name = f"Band {b + 1}"
            if meta.band_details and b < len(meta.band_details):
                b_name = meta.band_details[b].name or b_name
            item = QListWidgetItem(f"[{b + 1}] {b_name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Default check true-color bands or top 3 bands
            should_check = b in default_bands if b < 3 or b in default_bands else False
            item.setCheckState(Qt.Checked if should_check else Qt.Unchecked)
            item.setData(Qt.UserRole, b)
            self.ms_band_list.addItem(item)

    def _auto_detect_panchromatic(self) -> None:
        """Search if a 15m Band 8 GeoTIFF exists in the dataset directory."""
        lid = self.cb_ms_layers.currentData()
        if not lid or lid not in self.available_layers:
            self.rb_pan_layer.setChecked(True)
            return

        lyr, rdr = self.available_layers[lid]
        found_pan_file = None

        if hasattr(rdr, "directory") and os.path.isdir(rdr.directory):
            # Check for Landsat B8 file in folder
            for fname in os.listdir(rdr.directory):
                fl = fname.lower()
                if ("_b8.tif" in fl or "_b8.tiff" in fl or "_band8" in fl) and not fl.endswith(".ovr"):
                    found_pan_file = os.path.join(rdr.directory, fname)
                    break

        if found_pan_file:
            self.rb_pan_file.setChecked(True)
            self.txt_pan_file.setText(found_pan_file)
            self.txt_pan_file.setEnabled(True)
            self.btn_browse_pan.setEnabled(True)
            self.cb_pan_layers.setEnabled(False)
        else:
            self.rb_pan_layer.setChecked(True)
            self._on_pan_source_toggled(True)

    def _on_pan_source_toggled(self, checked: bool) -> None:
        is_layer = self.rb_pan_layer.isChecked()
        self.cb_pan_layers.setEnabled(is_layer)
        self.txt_pan_file.setEnabled(not is_layer)
        self.btn_browse_pan.setEnabled(not is_layer)

    def _browse_pan_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("dialog.pansharpen.browse_pan"),
            "",
            "GeoTIFF Raster (*.tif *.tiff *.img);;All Files (*)",
        )
        if path:
            self.txt_pan_file.setText(path)

    def _start_fusion(self) -> None:
        ms_lid = self.cb_ms_layers.currentData()
        if not ms_lid or ms_lid not in self.available_layers:
            QMessageBox.warning(self, tr("dialog.export.err_title"), "Invalid Multispectral Layer.")
            return

        ms_lyr, ms_rdr = self.available_layers[ms_lid]

        # Gather selected MS bands
        selected_bands = []
        for i in range(self.ms_band_list.count()):
            item = self.ms_band_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_bands.append(item.data(Qt.UserRole))

        if not selected_bands:
            QMessageBox.warning(self, tr("dialog.export.err_title"), tr("dialog.pansharpen.err_no_ms_bands"))
            return

        pan_reader = None
        pan_file = None

        if self.rb_pan_layer.isChecked():
            pan_lid = self.cb_pan_layers.currentData()
            if not pan_lid or pan_lid not in self.available_layers:
                QMessageBox.warning(self, tr("dialog.export.err_title"), "Invalid Panchromatic Layer.")
                return
            pan_lyr, pan_reader = self.available_layers[pan_lid]
        else:
            pan_file = self.txt_pan_file.text().strip()
            if not pan_file or not os.path.isfile(pan_file):
                QMessageBox.warning(self, tr("dialog.export.err_title"), tr("dialog.pansharpen.err_no_pan_file"))
                return

        out_name = self.txt_name.text().strip() or "PanSharpened_Image"
        method = self.cb_method.currentData()

        self.btn_fuse.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(selected_bands) * 2 + 2)

        self._worker = PanSharpenWorker(
            ms_reader=ms_rdr,
            selected_bands=selected_bands,
            pan_reader=pan_reader,
            pan_file_path=pan_file,
            method=method,
            name=out_name,
        )
        self._worker.progress.connect(self._on_fusion_progress)
        self._worker.finished.connect(self._on_fusion_finished)
        self._worker.failed.connect(self._on_fusion_failed)
        self._worker.start()

    def _on_fusion_progress(self, current: int, total: int) -> None:
        self.progress_bar.setValue(current)

    def _on_fusion_finished(self, name: str, result: np.ndarray, pan_meta: object) -> None:
        self.progress_bar.setVisible(False)
        self.btn_fuse.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        event_bus.status_message.emit(f"Pan-Sharpening completed: {name} (Shape: {result.shape})", 4000)
        self.result_generated.emit(name, result, pan_meta)
        self.accept()

    def _on_fusion_failed(self, error_msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_fuse.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        QMessageBox.critical(
            self,
            tr("dialog.export.err_title"),
            f"Pan-Sharpening Error:\n{error_msg}",
        )
