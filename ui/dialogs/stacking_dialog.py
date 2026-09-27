"""OpenENVI Layer Stacking Dialog.

Enables assembling multiple raster bands from open layers or disk files
into a unified multi-band dataset with custom ordering and metadata synthesis.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
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
    QWidget,
)

from core.algorithms.stacking import stack_bands
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.io.reader import open_raster
from core.models import RasterLayer, RasterMetadata


class StackingWorker(QThread):
    """Background worker for multi-band stacking."""

    progress = Signal(int, int)
    finished = Signal(str, np.ndarray, object)  # name, cube, metadata
    failed = Signal(str)

    def __init__(self, band_sources: List[Tuple[BaseRasterReader, int]], name: str):
        super().__init__()
        self.band_sources = band_sources
        self.name = name

    def run(self):
        try:
            cube, meta = stack_bands(
                self.band_sources,
                progress_callback=lambda c, t: self.progress.emit(c, t),
            )
            self.finished.emit(self.name, cube, meta)
        except Exception as e:
            self.failed.emit(str(e))


class LayerStackingDialog(QDialog):
    """Dialog for selecting, re-ordering, and combining bands into a new layer."""

    result_generated = Signal(str, np.ndarray, object)

    def __init__(
        self,
        layers: Dict[str, Tuple[RasterLayer, BaseRasterReader]],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.layers = layers
        self._external_readers: List[BaseRasterReader] = []

        self.setWindowTitle(tr("toolbox.tool_stacking"))
        self.resize(750, 500)

        self._init_ui()
        self._populate_available_bands()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 2-column selection list layout
        lists_layout = QHBoxLayout()

        # Left: Available Bands
        grp_avail = QGroupBox(tr("stacking.grp_avail"))
        layout_avail = QVBoxLayout(grp_avail)
        self.list_avail = QListWidget()
        layout_avail.addWidget(self.list_avail)

        btn_bar_avail = QHBoxLayout()
        self.btn_browse = QPushButton(tr("stacking.btn_browse"))
        self.btn_browse.clicked.connect(self._browse_external_file)
        btn_bar_avail.addWidget(self.btn_browse)

        self.btn_add_selected = QPushButton(tr("stacking.btn_add_selected"))
        self.btn_add_selected.clicked.connect(self._add_selected_bands)
        btn_bar_avail.addWidget(self.btn_add_selected)

        layout_avail.addLayout(btn_bar_avail)
        lists_layout.addWidget(grp_avail, stretch=1)

        # Right: Selected Bands for Output
        grp_selected = QGroupBox(tr("stacking.grp_selected"))
        layout_selected = QVBoxLayout(grp_selected)
        self.list_selected = QListWidget()
        layout_selected.addWidget(self.list_selected)

        order_bar = QHBoxLayout()
        self.btn_up = QPushButton(tr("stacking.btn_up"))
        self.btn_up.clicked.connect(self._move_up)
        order_bar.addWidget(self.btn_up)

        self.btn_down = QPushButton(tr("stacking.btn_down"))
        self.btn_down.clicked.connect(self._move_down)
        order_bar.addWidget(self.btn_down)

        self.btn_remove = QPushButton(tr("stacking.btn_remove"))
        self.btn_remove.clicked.connect(self._remove_selected)
        order_bar.addWidget(self.btn_remove)

        self.btn_clear = QPushButton(tr("stacking.btn_clear"))
        self.btn_clear.clicked.connect(self.list_selected.clear)
        order_bar.addWidget(self.btn_clear)

        layout_selected.addLayout(order_bar)
        lists_layout.addWidget(grp_selected, stretch=1)

        layout.addLayout(lists_layout, stretch=1)

        # Output options
        out_bar = QHBoxLayout()
        out_bar.addWidget(QLabel(tr("stacking.out_name")))
        self.txt_out_name = QLineEdit("Layer_Stacked")
        out_bar.addWidget(self.txt_out_name)
        layout.addLayout(out_bar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Action buttons
        btn_actions = QHBoxLayout()
        btn_actions.addStretch()

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_actions.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton(tr("dialog.btn_ok"))
        self.btn_ok.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_ok.clicked.connect(self._start_stacking)
        btn_actions.addWidget(self.btn_ok)

        layout.addLayout(btn_actions)

    def _populate_available_bands(self):
        for lid, (layer, reader) in self.layers.items():
            meta = layer.metadata
            for b in range(meta.bands):
                b_name = (
                    meta.band_details[b].name
                    if meta.band_details and b < len(meta.band_details) and meta.band_details[b].name
                    else f"Band {b + 1}"
                )
                item = QListWidgetItem(f"[{layer.name}] {b_name}")
                item.setData(Qt.UserRole, (reader, b, f"[{layer.name}] {b_name}"))
                self.list_avail.addItem(item)

    def _browse_external_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Raster File to Add Bands",
            "",
            "Remote Sensing Rasters (*.tif *.tiff *.dat *.img);;All Files (*)",
        )
        if not file_path:
            return

        try:
            reader = open_raster(file_path)
            self._external_readers.append(reader)
            fname = reader.metadata.filename or "File"
            for b in range(reader.metadata.bands):
                b_name = (
                    reader.metadata.band_details[b].name
                    if reader.metadata.band_details and b < len(reader.metadata.band_details) and reader.metadata.band_details[b].name
                    else f"Band {b + 1}"
                )
                item = QListWidgetItem(f"[{fname}] {b_name}")
                item.setData(Qt.UserRole, (reader, b, f"[{fname}] {b_name}"))
                self.list_avail.addItem(item)
        except Exception as e:
            QMessageBox.critical(self, tr("dialog.error"), f"{tr('stacking.err_open')}: {e}")

    def _add_selected_bands(self):
        for item in self.list_avail.selectedItems():
            data = item.data(Qt.UserRole)
            new_item = QListWidgetItem(item.text())
            new_item.setData(Qt.UserRole, data)
            self.list_selected.addItem(new_item)

    def _remove_selected(self):
        for item in self.list_selected.selectedItems():
            row = self.list_selected.row(item)
            self.list_selected.takeItem(row)

    def _move_up(self):
        row = self.list_selected.currentRow()
        if row > 0:
            item = self.list_selected.takeItem(row)
            self.list_selected.insertItem(row - 1, item)
            self.list_selected.setCurrentRow(row - 1)

    def _move_down(self):
        row = self.list_selected.currentRow()
        if 0 <= row < self.list_selected.count() - 1:
            item = self.list_selected.takeItem(row)
            self.list_selected.insertItem(row + 1, item)
            self.list_selected.setCurrentRow(row + 1)

    def _start_stacking(self):
        if self.list_selected.count() == 0:
            QMessageBox.warning(
                self,
                tr("stacking.err_no_bands_title"),
                tr("stacking.err_no_bands_msg"),
            )
            return

        sources = []
        for i in range(self.list_selected.count()):
            item = self.list_selected.item(i)
            reader, b_idx, _ = item.data(Qt.UserRole)
            sources.append((reader, b_idx))

        name = self.txt_out_name.text().strip() or "Stacked_Raster"

        self.btn_ok.setEnabled(False)
        self.progress_bar.setRange(0, len(sources))
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.worker = StackingWorker(sources, name)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_stacking_finished)
        self.worker.failed.connect(self._on_stacking_failed)
        self.worker.start()

    def _on_stacking_finished(self, name: str, cube: np.ndarray, meta: object):
        self.result_generated.emit(name, cube, meta)
        self.accept()

    def _on_stacking_failed(self, err: str):
        self.btn_ok.setEnabled(True)
        self.progress_bar.hide()
        QMessageBox.critical(self, tr("dialog.error"), f"{tr('stacking.err_failed')}: {err}")

    def closeEvent(self, event):
        for r in self._external_readers:
            try:
                r.close()
            except Exception:
                pass
        super().closeEvent(event)
