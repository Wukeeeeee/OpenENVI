"""OpenENVI Radiometric Calibration & Atmospheric Correction Dialog.

Converts raw satellite Digital Numbers (DN) to TOA Reflectance, TOA Radiance,
or DOS-corrected Surface Reflectance with auto-detected Landsat MTL parameters.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
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
    QRadioButton,
    QVBoxLayout,
)

from core.algorithms.radiometry import execute_calibration, extract_landsat_cal_params
from core.events import event_bus
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import RasterLayer


class RadiometryWorker(QThread):
    """Background worker for quantitative calibration calculation."""

    progress = Signal(int, int)
    finished = Signal(str, np.ndarray, object)  # name, result_array, parent_meta
    failed = Signal(str)

    def __init__(
        self,
        reader: BaseRasterReader,
        cal_type: str,
        band_indices: List[int],
        name: str,
        custom_mult: Optional[float] = None,
        custom_add: Optional[float] = None,
        custom_sun_elev: Optional[float] = None,
    ):
        super().__init__()
        self.reader = reader
        self.cal_type = cal_type
        self.band_indices = band_indices
        self.name = name
        self.custom_mult = custom_mult
        self.custom_add = custom_add
        self.custom_sun_elev = custom_sun_elev

    def run(self):
        try:
            result = execute_calibration(
                reader=self.reader,
                cal_type=self.cal_type,
                band_indices=self.band_indices,
                custom_mult=self.custom_mult,
                custom_add=self.custom_add,
                custom_sun_elev=self.custom_sun_elev,
                progress_callback=lambda s, t: self.progress.emit(s, t),
            )
            self.finished.emit(self.name, result, self.reader.metadata)
        except Exception as e:
            self.failed.emit(str(e))


class RadiometryDialog(QDialog):
    """Dialog for radiometric calibration and atmospheric correction."""

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
        self._worker: Optional[RadiometryWorker] = None

        self.setWindowTitle(tr("dialog.radiometry.title"))
        self.resize(540, 600)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # 1. Layer Selection
        grp_layer = QGroupBox(tr("dialog.export.grp_layer"))
        lay_layer = QHBoxLayout(grp_layer)
        self.cb_layers = QComboBox()
        for lid, (lyr, _) in self.available_layers.items():
            self.cb_layers.addItem(lyr.name, lid)
            if lid == self.active_layer_id:
                self.cb_layers.setCurrentIndex(self.cb_layers.count() - 1)
        self.cb_layers.currentIndexChanged.connect(self._on_layer_changed)
        lay_layer.addWidget(self.cb_layers)
        main_layout.addWidget(grp_layer)

        # 2. Calibration Type
        grp_type = QGroupBox(tr("dialog.radiometry.grp_type"))
        lay_type = QVBoxLayout(grp_type)

        self.bg_type = QButtonGroup(self)
        self.rb_toa_refl = QRadioButton(tr("dialog.radiometry.type_toa_refl"))
        self.rb_dos_refl = QRadioButton(tr("dialog.radiometry.type_dos_refl"))
        self.rb_radiance = QRadioButton(tr("dialog.radiometry.type_radiance"))

        self.bg_type.addButton(self.rb_toa_refl)
        self.bg_type.addButton(self.rb_dos_refl)
        self.bg_type.addButton(self.rb_radiance)

        self.rb_toa_refl.setChecked(True)
        lay_type.addWidget(self.rb_toa_refl)
        lay_type.addWidget(self.rb_dos_refl)
        lay_type.addWidget(self.rb_radiance)
        main_layout.addWidget(grp_type)

        # 3. Band Selection
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

        # 4. Parameters Notice / Custom Parameters
        grp_params = QGroupBox(tr("dialog.radiometry.grp_params"))
        lay_params = QVBoxLayout(grp_params)

        self.lbl_mtl_info = QLabel()
        self.lbl_mtl_info.setWordWrap(True)
        lay_params.addWidget(self.lbl_mtl_info)

        # Custom fallback spinboxes
        self.custom_container = QVBoxLayout()
        row_mult = QHBoxLayout()
        row_mult.addWidget(QLabel(tr("dialog.radiometry.multiplier")))
        self.spin_mult = QDoubleSpinBox()
        self.spin_mult.setRange(-1e6, 1e6)
        self.spin_mult.setDecimals(6)
        self.spin_mult.setValue(0.00002)
        row_mult.addWidget(self.spin_mult)
        self.custom_container.addLayout(row_mult)

        row_add = QHBoxLayout()
        row_add.addWidget(QLabel(tr("dialog.radiometry.offset")))
        self.spin_add = QDoubleSpinBox()
        self.spin_add.setRange(-1e6, 1e6)
        self.spin_add.setDecimals(6)
        self.spin_add.setValue(-0.1)
        row_add.addWidget(self.spin_add)
        self.custom_container.addLayout(row_add)

        row_sun = QHBoxLayout()
        row_sun.addWidget(QLabel(tr("dialog.radiometry.sun_elevation")))
        self.spin_sun = QDoubleSpinBox()
        self.spin_sun.setRange(1.0, 90.0)
        self.spin_sun.setDecimals(2)
        self.spin_sun.setValue(45.0)
        row_sun.addWidget(self.spin_sun)
        self.custom_container.addLayout(row_sun)

        lay_params.addLayout(self.custom_container)
        main_layout.addWidget(grp_params)

        # 5. Output Name
        row_name = QHBoxLayout()
        row_name.addWidget(QLabel(tr("dialog.pansharpen.out_name")))
        self.txt_name = QLineEdit()
        self.txt_name.setText("Calibrated_Reflectance")
        row_name.addWidget(self.txt_name)
        main_layout.addLayout(row_name)

        # 6. Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 7. Action Buttons
        btn_bar = QHBoxLayout()
        self.btn_run = QPushButton(tr("dialog.radiometry.btn_calibrate"))
        self.btn_run.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_run.clicked.connect(self._start_calibration)
        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addStretch()
        btn_bar.addWidget(self.btn_run)
        btn_bar.addWidget(self.btn_cancel)
        main_layout.addLayout(btn_bar)

        # Initialize
        self._update_layer_state()

    def _on_layer_changed(self) -> None:
        self._update_layer_state()

    def _update_layer_state(self) -> None:
        lid = self.cb_layers.currentData()
        if not lid or lid not in self.available_layers:
            return

        lyr, rdr = self.available_layers[lid]
        meta = rdr.metadata

        # Populate bands
        self.band_list.clear()
        for b in range(meta.bands):
            b_name = f"Band {b + 1}"
            if meta.band_details and b < len(meta.band_details):
                b_name = meta.band_details[b].name or b_name
            item = QListWidgetItem(f"[{b + 1}] {b_name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, b)
            self.band_list.addItem(item)

        # Check MTL params
        cal_params = extract_landsat_cal_params(rdr)
        if cal_params:
            sun_elev = cal_params["sun_elevation"]
            self.lbl_mtl_info.setText(
                f"{tr('dialog.radiometry.mtl_detected')} (Sun Elevation: {sun_elev:.2f}°)"
            )
            self.lbl_mtl_info.setStyleSheet("color: #2ecc71; font-weight: bold; padding: 4px;")
            self.spin_sun.setValue(sun_elev)
            # Hide custom gain/offset controls when MTL metadata provides exact values
            self._set_custom_controls_visible(False)
        else:
            self.lbl_mtl_info.setText(tr("dialog.radiometry.custom_notice"))
            self.lbl_mtl_info.setStyleSheet("color: #f39c12; font-weight: bold; padding: 4px;")
            self._set_custom_controls_visible(True)

    def _set_custom_controls_visible(self, visible: bool) -> None:
        for i in range(self.custom_container.count()):
            layout_item = self.custom_container.itemAt(i)
            if layout_item.layout():
                for j in range(layout_item.layout().count()):
                    w = layout_item.layout().itemAt(j).widget()
                    if w:
                        w.setVisible(visible)

    def _select_all_bands(self) -> None:
        for i in range(self.band_list.count()):
            self.band_list.item(i).setCheckState(Qt.Checked)

    def _clear_all_bands(self) -> None:
        for i in range(self.band_list.count()):
            self.band_list.item(i).setCheckState(Qt.Unchecked)

    def _start_calibration(self) -> None:
        lid = self.cb_layers.currentData()
        if not lid or lid not in self.available_layers:
            return

        lyr, rdr = self.available_layers[lid]

        # Gather selected bands
        selected_bands = []
        for i in range(self.band_list.count()):
            item = self.band_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_bands.append(item.data(Qt.UserRole))

        if not selected_bands:
            QMessageBox.warning(self, tr("dialog.export.err_title"), tr("dialog.export.err_no_bands"))
            return

        if self.rb_toa_refl.isChecked():
            cal_type = "reflectance"
        elif self.rb_dos_refl.isChecked():
            cal_type = "dos"
        else:
            cal_type = "radiance"

        name = self.txt_name.text().strip() or f"Calibrated_{cal_type}"
        custom_mult = self.spin_mult.value()
        custom_add = self.spin_add.value()
        custom_sun_elev = self.spin_sun.value()

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(selected_bands))

        self._worker = RadiometryWorker(
            reader=rdr,
            cal_type=cal_type,
            band_indices=selected_bands,
            name=name,
            custom_mult=custom_mult,
            custom_add=custom_add,
            custom_sun_elev=custom_sun_elev,
        )
        self._worker.progress.connect(self._on_cal_progress)
        self._worker.finished.connect(self._on_cal_finished)
        self._worker.failed.connect(self._on_cal_failed)
        self._worker.start()

    def _on_cal_progress(self, current: int, total: int) -> None:
        self.progress_bar.setValue(current)

    def _on_cal_finished(self, name: str, result: np.ndarray, parent_meta: object) -> None:
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        event_bus.status_message.emit(f"Calibration completed: {name}", 4000)
        self.result_generated.emit(name, result, parent_meta)
        self.accept()

    def _on_cal_failed(self, error_msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        QMessageBox.critical(
            self,
            tr("dialog.export.err_title"),
            f"Calibration Error:\n{error_msg}",
        )
