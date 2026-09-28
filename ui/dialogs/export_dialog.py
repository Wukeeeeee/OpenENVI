"""OpenENVI Raster Export Dialog.

Allows exporting in-memory derived layers or persistent datasets into GeoTIFF or ENVI
formats with custom band selection, data type conversion, and compression options.
"""

import os
from typing import Dict, List, Optional, Tuple
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
    QVBoxLayout,
)

from core.events import event_bus
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.io.writer import export_raster
from core.models import RasterLayer


class ExportWorker(QThread):
    """Background thread worker for streaming raster export."""

    progress = Signal(int, int)  # current, total
    finished = Signal(str)       # output_path
    failed = Signal(str)         # error_message

    def __init__(
        self,
        reader: BaseRasterReader,
        output_path: str,
        fmt: str,
        band_indices: List[int],
        dtype: Optional[str],
        compress: Optional[str],
    ):
        super().__init__()
        self.reader = reader
        self.output_path = output_path
        self.fmt = fmt
        self.band_indices = band_indices
        self.dtype = dtype
        self.compress = compress

    def run(self):
        try:
            res = export_raster(
                reader=self.reader,
                output_path=self.output_path,
                format=self.fmt,
                band_indices=self.band_indices,
                dtype=self.dtype,
                compress=self.compress,
                progress_callback=lambda c, t: self.progress.emit(c, t),
            )
            self.finished.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


class ExportRasterDialog(QDialog):
    """Dialog for exporting a raster layer to GeoTIFF or ENVI file formats."""

    export_completed = Signal(str)

    def __init__(
        self,
        layer: RasterLayer,
        reader: BaseRasterReader,
        available_layers: Optional[Dict[str, Tuple[RasterLayer, BaseRasterReader]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.current_layer = layer
        self.current_reader = reader
        self.available_layers = available_layers or {layer.layer_id: (layer, reader)}
        self._worker: Optional[ExportWorker] = None

        self.setWindowTitle(tr("dialog.export.title"))
        self.resize(520, 560)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # 1. Layer Selection
        grp_layer = QGroupBox(tr("dialog.export.grp_layer"))
        lay_layer = QHBoxLayout(grp_layer)
        self.cb_layers = QComboBox()
        for lid, (lyr, rdr) in self.available_layers.items():
            self.cb_layers.addItem(lyr.name, lid)
            if lid == self.current_layer.layer_id:
                self.cb_layers.setCurrentIndex(self.cb_layers.count() - 1)
        self.cb_layers.currentIndexChanged.connect(self._on_layer_selection_changed)
        lay_layer.addWidget(self.cb_layers)
        main_layout.addWidget(grp_layer)

        # 2. Band Selection
        grp_bands = QGroupBox(tr("dialog.export.grp_bands"))
        lay_bands = QVBoxLayout(grp_bands)

        btn_box = QHBoxLayout()
        self.btn_select_all = QPushButton(tr("dialog.export.btn_select_all"))
        self.btn_select_all.clicked.connect(self._select_all_bands)
        self.btn_clear_all = QPushButton(tr("dialog.export.btn_clear_all"))
        self.btn_clear_all.clicked.connect(self._clear_all_bands)
        btn_box.addWidget(self.btn_select_all)
        btn_box.addWidget(self.btn_clear_all)
        btn_box.addStretch()
        lay_bands.addLayout(btn_box)

        self.band_list = QListWidget()
        lay_bands.addWidget(self.band_list)
        main_layout.addWidget(grp_bands)

        # 3. Export Parameters
        grp_params = QGroupBox(tr("dialog.export.grp_params"))
        lay_params = QVBoxLayout(grp_params)

        # Format
        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel(tr("dialog.export.format")))
        self.cb_format = QComboBox()
        self.cb_format.addItem("GeoTIFF (*.tif)", "GTiff")
        self.cb_format.addItem("ENVI Standard (*.dat / *.hdr)", "ENVI")
        self.cb_format.currentIndexChanged.connect(self._on_format_changed)
        row_fmt.addWidget(self.cb_format)
        lay_params.addLayout(row_fmt)

        # Data Type
        row_dtype = QHBoxLayout()
        row_dtype.addWidget(QLabel(tr("dialog.export.dtype")))
        self.cb_dtype = QComboBox()
        self.cb_dtype.addItem(tr("dialog.export.dtype_auto"), "")
        self.cb_dtype.addItem("Float32 (32-bit Float)", "float32")
        self.cb_dtype.addItem("UInt16 (16-bit Unsigned)", "uint16")
        self.cb_dtype.addItem("UInt8 / Byte (8-bit)", "uint8")
        self.cb_dtype.addItem("Int16 (16-bit Signed)", "int16")
        row_dtype.addWidget(self.cb_dtype)
        lay_params.addLayout(row_dtype)

        # Compression (for GeoTIFF)
        self.row_compress = QHBoxLayout()
        self.lbl_compress = QLabel(tr("dialog.export.compress"))
        self.cb_compress = QComboBox()
        self.cb_compress.addItem("LZW", "lzw")
        self.cb_compress.addItem("DEFLATE", "deflate")
        self.cb_compress.addItem(tr("dialog.export.compress_none"), "none")
        self.row_compress.addWidget(self.lbl_compress)
        self.row_compress.addWidget(self.cb_compress)
        lay_params.addLayout(self.row_compress)

        main_layout.addWidget(grp_params)

        # 4. Destination File
        grp_dest = QGroupBox(tr("dialog.export.grp_dest"))
        lay_dest = QHBoxLayout(grp_dest)
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText(tr("dialog.export.path_placeholder"))
        self.btn_browse = QPushButton(tr("dialog.export.btn_browse"))
        self.btn_browse.clicked.connect(self._browse_save_path)
        lay_dest.addWidget(self.txt_path)
        lay_dest.addWidget(self.btn_browse)
        main_layout.addWidget(grp_dest)

        # Autoload checkbox
        self.chk_autoload = QCheckBox(tr("dialog.export.chk_autoload") if tr("dialog.export.chk_autoload") != "dialog.export.chk_autoload" else "Load exported raster into OpenENVI upon completion")
        self.chk_autoload.setChecked(True)
        main_layout.addWidget(self.chk_autoload)

        # 5. Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 6. Action Buttons
        btn_bar = QHBoxLayout()
        self.btn_export = QPushButton(tr("dialog.export.btn_export"))
        self.btn_export.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_export.clicked.connect(self._start_export)
        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addStretch()
        btn_bar.addWidget(self.btn_export)
        btn_bar.addWidget(self.btn_cancel)
        main_layout.addLayout(btn_bar)

        # Initialize band list and default destination
        self._populate_band_list()
        self._init_default_path()

    def _on_layer_selection_changed(self, index: int) -> None:
        """Handle layer combo selection change."""
        lid = self.cb_layers.itemData(index)
        if lid in self.available_layers:
            self.current_layer, self.current_reader = self.available_layers[lid]
            self._populate_band_list()
            self._init_default_path()

    def _populate_band_list(self) -> None:
        """Populate the band checklist according to the active reader."""
        self.band_list.clear()
        meta = self.current_reader.metadata
        for b in range(meta.bands):
            b_name = f"Band {b + 1}"
            if meta.band_details and b < len(meta.band_details):
                b_info = meta.band_details[b]
                if b_info.wavelength is not None:
                    if b_info.name:
                        b_name = f"{b_info.name} ({b_info.wavelength:.1f} {b_info.wavelength_unit})"
                    else:
                        b_name = f"Band {b + 1} ({b_info.wavelength:.1f} {b_info.wavelength_unit})"
                elif b_info.name:
                    b_name = b_info.name
            item = QListWidgetItem(f"[{b + 1}] {b_name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, b)
            self.band_list.addItem(item)

    def _select_all_bands(self) -> None:
        for i in range(self.band_list.count()):
            self.band_list.item(i).setCheckState(Qt.Checked)

    def _clear_all_bands(self) -> None:
        for i in range(self.band_list.count()):
            self.band_list.item(i).setCheckState(Qt.Unchecked)

    def _on_format_changed(self, index: int) -> None:
        """Toggle compression availability and update extension."""
        is_gtiff = self.cb_format.currentData() == "GTiff"
        self.lbl_compress.setEnabled(is_gtiff)
        self.cb_compress.setEnabled(is_gtiff)

        cur_path = self.txt_path.text().strip()
        if cur_path:
            base, _ = os.path.splitext(cur_path)
            new_ext = ".tif" if is_gtiff else ".dat"
            self.txt_path.setText(base + new_ext)

    def _init_default_path(self) -> None:
        """Set a default destination path in user's temp/working directory."""
        clean_name = self.current_layer.name.replace(" ", "_").replace("/", "_")
        is_gtiff = self.cb_format.currentData() == "GTiff"
        ext = ".tif" if is_gtiff else ".dat"
        default_dir = os.path.abspath("E:/temp") if os.path.exists("E:/temp") else os.path.expanduser("~")
        self.txt_path.setText(os.path.join(default_dir, f"{clean_name}_export{ext}"))

    def _browse_save_path(self) -> None:
        """Open file dialog to pick export destination."""
        is_gtiff = self.cb_format.currentData() == "GTiff"
        if is_gtiff:
            filter_str = "GeoTIFF (*.tif *.tiff);;All Files (*)"
        else:
            filter_str = "ENVI Binary (*.dat *.raw *.img);;All Files (*)"

        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("dialog.export.save_dialog_title"),
            self.txt_path.text().strip(),
            filter_str,
        )
        if path:
            self.txt_path.setText(path)

    def _start_export(self) -> None:
        """Validate inputs and launch background export thread."""
        target_path = self.txt_path.text().strip()
        if not target_path:
            QMessageBox.warning(self, tr("dialog.export.err_title"), tr("dialog.export.err_no_path"))
            return

        # Gather selected bands
        selected_bands = []
        for i in range(self.band_list.count()):
            item = self.band_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_bands.append(item.data(Qt.UserRole))

        if not selected_bands:
            QMessageBox.warning(self, tr("dialog.export.err_title"), tr("dialog.export.err_no_bands"))
            return

        fmt = self.cb_format.currentData()
        dtype = self.cb_dtype.currentData() or None
        compress = self.cb_compress.currentData() if fmt == "GTiff" else None

        self.btn_export.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(selected_bands))

        self._worker = ExportWorker(
            reader=self.current_reader,
            output_path=target_path,
            fmt=fmt,
            band_indices=selected_bands,
            dtype=dtype,
            compress=compress,
        )
        self._worker.progress.connect(self._on_export_progress)
        self._worker.finished.connect(self._on_export_finished)
        self._worker.failed.connect(self._on_export_failed)
        self._worker.start()

    def _on_export_progress(self, current: int, total: int) -> None:
        self.progress_bar.setValue(current)

    def _on_export_finished(self, out_path: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        event_bus.status_message.emit(f"Raster exported to: {out_path}", 4000)
        if self.chk_autoload.isChecked():
            self.export_completed.emit(out_path)
        else:
            QMessageBox.information(
                self,
                tr("dialog.export.success_title"),
                f"{tr('dialog.export.success_msg')}\n\n{out_path}",
            )
        self.accept()

    def _on_export_failed(self, error_msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        QMessageBox.critical(
            self,
            tr("dialog.export.err_title"),
            f"{tr('dialog.export.err_export_failed')}:\n{error_msg}",
        )
