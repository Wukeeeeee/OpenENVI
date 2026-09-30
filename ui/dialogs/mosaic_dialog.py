"""OpenENVI Mosaic Dialog.

Combines the loaded georeferenced layers onto one grid. Inputs are resampled onto a
union grid built at the reference raster's resolution, and overlapping regions are
blended with an optional feather width so seams do not appear as hard edges.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.algorithms.mosaic import build_mosaic_grid, mosaic_rasters
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import RasterLayer, RasterMetadata


def _resolution_text(reader: BaseRasterReader) -> str:
    """Pixel size as text for the inputs table, or a dash when unavailable."""
    transform = reader.metadata.transform
    if transform is None:
        return "-"
    values = tuple(transform)[:6]
    return f"{values[0]:g} x {abs(values[4]):g}"


class MosaicWorker(QThread):
    """Background worker that mosaics the selected readers."""

    progress = Signal(int, int)
    finished = Signal(str, np.ndarray, object)
    failed = Signal(str)

    def __init__(
        self,
        readers: List[BaseRasterReader],
        name: str,
        resample_method: str,
        background_value: float,
        feather_pixels: float,
    ):
        super().__init__()
        self.readers = readers
        self.name = name
        self.resample_method = resample_method
        self.background_value = background_value
        self.feather_pixels = feather_pixels

    def run(self):
        try:
            cube, meta = mosaic_rasters(
                self.readers,
                resample_method=self.resample_method,
                background_value=self.background_value,
                feather_pixels=self.feather_pixels,
                progress_callback=lambda c, t: self.progress.emit(c, t),
            )
            self.finished.emit(self.name, cube, meta)
        except Exception as e:  # surfaced in the dialog rather than killing the thread
            self.failed.emit(str(e))


class MosaicDialog(QDialog):
    """Dialog for mosaicking loaded layers into one raster."""

    result_generated = Signal(str, np.ndarray, object)

    def __init__(
        self,
        layers: Dict[str, Tuple[RasterLayer, BaseRasterReader]],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.layers = layers
        self._worker: Optional[MosaicWorker] = None

        self.setWindowTitle(tr("toolbox.tool_mosaic"))
        self.resize(760, 560)

        self._init_ui()
        self._populate_inputs()

    # ------------------------------------------------------------------ UI

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        grp_inputs = QGroupBox(tr("mosaic.grp_inputs"))
        l_inputs = QVBoxLayout(grp_inputs)

        self.table_inputs = QTableWidget(0, 5)
        self.table_inputs.setHorizontalHeaderLabels([
            tr("mosaic.col_use"),
            tr("mosaic.col_name"),
            tr("mosaic.col_size"),
            tr("mosaic.col_bands"),
            tr("mosaic.col_crs"),
        ])
        self.table_inputs.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_inputs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_inputs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        l_inputs.addWidget(self.table_inputs)

        h_sel = QHBoxLayout()
        self.btn_all = QPushButton(tr("mosaic.select_all"))
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        h_sel.addWidget(self.btn_all)
        self.btn_none = QPushButton(tr("mosaic.select_none"))
        self.btn_none.clicked.connect(lambda: self._set_all(False))
        h_sel.addWidget(self.btn_none)
        h_sel.addStretch()
        self.lbl_preview = QLabel("")
        h_sel.addWidget(self.lbl_preview)
        l_inputs.addLayout(h_sel)

        layout.addWidget(grp_inputs, stretch=1)

        # Output grid preview
        grp_grid = QGroupBox(tr("mosaic.grp_grid"))
        l_grid = QVBoxLayout(grp_grid)
        self.lbl_grid = QLabel("-")
        self.lbl_grid.setWordWrap(True)
        l_grid.addWidget(self.lbl_grid)
        layout.addWidget(grp_grid)

        # Parameters
        grp_params = QGroupBox(tr("mosaic.grp_params"))
        l_params = QVBoxLayout(grp_params)

        h_resample = QHBoxLayout()
        h_resample.addWidget(QLabel(tr("mosaic.lbl_resample")))
        self.cmb_resample = QComboBox()
        for key in ("bilinear", "nearest", "cubic", "cubic_spline", "average", "lanczos"):
            self.cmb_resample.addItem(tr(f"mosaic.resample_{key}"), key)
        self.cmb_resample.setToolTip(tr("mosaic.tip_resample"))
        h_resample.addWidget(self.cmb_resample, stretch=1)
        l_params.addLayout(h_resample)

        h_feather = QHBoxLayout()
        h_feather.addWidget(QLabel(tr("mosaic.lbl_feather")))
        self.spin_feather = QDoubleSpinBox()
        self.spin_feather.setRange(0.0, 500.0)
        self.spin_feather.setDecimals(1)
        self.spin_feather.setValue(0.0)
        self.spin_feather.setSuffix(" px")
        self.spin_feather.setToolTip(tr("mosaic.tip_feather"))
        self.spin_feather.valueChanged.connect(self._update_grid_preview)
        h_feather.addWidget(self.spin_feather, stretch=1)
        l_params.addLayout(h_feather)

        h_bg = QHBoxLayout()
        h_bg.addWidget(QLabel(tr("mosaic.lbl_background")))
        self.spin_background = QDoubleSpinBox()
        self.spin_background.setRange(-1e9, 1e9)
        self.spin_background.setDecimals(4)
        self.spin_background.setValue(0.0)
        self.spin_background.setToolTip(tr("mosaic.tip_background"))
        h_bg.addWidget(self.spin_background, stretch=1)
        l_params.addLayout(h_bg)

        layout.addWidget(grp_params)

        # Output name
        h_out = QHBoxLayout()
        h_out.addWidget(QLabel(tr("mosaic.out_name")))
        self.txt_out_name = QLineEdit(tr("mosaic.default_name"))
        h_out.addWidget(self.txt_out_name)
        layout.addLayout(h_out)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

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

    # ------------------------------------------------------------- inputs

    def _populate_inputs(self):
        """Fill the inputs table with every georeferenced layer that loaded."""
        self.table_inputs.setRowCount(0)
        row = 0
        for _layer_id, (_layer, reader) in self.layers.items():
            meta = reader.metadata
            self.table_inputs.insertRow(row)

            item = QTableWidgetItem()
            # A raster without a geotransform cannot be mosaicked, so it is listed
            # but cannot be selected.
            usable = meta.transform is not None and meta.crs is not None
            item.setCheckState(Qt.Checked if usable else Qt.Unchecked)
            if not usable:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            item.setData(Qt.UserRole, _layer_id)
            self.table_inputs.setItem(row, 0, item)

            self.table_inputs.setItem(row, 1, QTableWidgetItem(_layer.name))
            self.table_inputs.setItem(
                row, 2,
                QTableWidgetItem(f"{meta.width} x {meta.height} @ {_resolution_text(reader)}"),
            )
            self.table_inputs.setItem(row, 3, QTableWidgetItem(str(meta.bands)))
            self.table_inputs.setItem(
                row, 4, QTableWidgetItem(str(meta.crs) if meta.crs else tr("mosaic.no_crs"))
            )
            row += 1

        self.btn_ok.setEnabled(row > 0)
        self._update_grid_preview()

    def _set_all(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.table_inputs.rowCount()):
            item = self.table_inputs.item(row, 0)
            if item.flags() & Qt.ItemIsEnabled:
                item.setCheckState(state)
        self._update_grid_preview()

    def selected_readers(self) -> List[BaseRasterReader]:
        """Readers for the checked rows, in table order."""
        readers = []
        for row in range(self.table_inputs.rowCount()):
            item = self.table_inputs.item(row, 0)
            if item.checkState() == Qt.Checked:
                readers.append(self.layers[item.data(Qt.UserRole)][1])
        return readers

    def _update_grid_preview(self):
        """Describe the union grid that the current selection would produce."""
        readers = self.selected_readers()
        if not readers:
            self.lbl_grid.setText(tr("mosaic.no_inputs"))
            self.btn_ok.setEnabled(False)
            return
        self.btn_ok.setEnabled(True)
        if len(readers) == 1:
            meta = readers[0].metadata
            self.lbl_grid.setText(
                tr("mosaic.grid_single").format(
                    w=meta.width, h=meta.height, res=_resolution_text(readers[0]), crs=meta.crs
                )
            )
            return
        try:
            transform, width, height, crs = build_mosaic_grid(readers)
        except Exception as e:
            self.lbl_grid.setText(tr("mosaic.grid_error").format(err=e))
            self.btn_ok.setEnabled(False)
            return
        feather = self.spin_feather.value()
        note = ""
        if feather > 0:
            note = " " + tr("mosaic.grid_feathered").format(feather=feather)
        self.lbl_grid.setText(
            tr("mosaic.grid_union").format(
                w=width, h=height, res=f"{transform.a:g} x {abs(transform.e):g}",
                crs=crs, n=len(readers),
            ) + note
        )

    # ------------------------------------------------------------- actions

    def _start(self):
        readers = self.selected_readers()
        if not readers:
            QMessageBox.warning(self, tr("dialog.error"), tr("mosaic.err_no_inputs"))
            return

        band_counts = {r.metadata.bands for r in readers}
        if len(band_counts) != 1:
            QMessageBox.warning(
                self, tr("dialog.error"),
                tr("mosaic.err_bands").format(counts=sorted(band_counts)),
            )
            return

        name = self.txt_out_name.text().strip() or tr("mosaic.default_name")
        self.btn_ok.setEnabled(False)
        self.progress_bar.setRange(0, 0)   # indeterminate: work is per band per source
        self.progress_bar.show()

        self._worker = MosaicWorker(
            readers=readers,
            name=name,
            resample_method=self.cmb_resample.currentData(),
            background_value=self.spin_background.value(),
            feather_pixels=self.spin_feather.value(),
        )
        self._worker.progress.connect(
            lambda c, t: self.progress_bar.setValue(c)
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_finished(self, name: str, cube: np.ndarray, meta: RasterMetadata):
        self.progress_bar.hide()
        self.btn_ok.setEnabled(True)
        self.result_generated.emit(name, cube, meta)
        self.accept()

    def _on_failed(self, err: str):
        self.progress_bar.hide()
        self.btn_ok.setEnabled(True)
        QMessageBox.critical(self, tr("dialog.error"), f"{tr('mosaic.err_failed')}: {err}")
