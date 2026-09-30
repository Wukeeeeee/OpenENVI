"""OpenENVI Spectral Information Divergence (SID) Dialog.

Implements supervised Spectral Information Divergence classification using reference
endmember spectra derived from Regions of Interest (ROIs) based on Kullback-Leibler divergence.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.algorithms.classification import create_thematic_rgb
from core.algorithms.spectral import spectral_information_divergence
from core.events import event_bus
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterLayer, RasterMetadata
from core.roi import ROI


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color string like #ff5500 to RGB tuple (255, 85, 0)."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        return (
            int(hex_str[0:2], 16),
            int(hex_str[2:4], 16),
            int(hex_str[4:6], 16),
        )
    return (255, 0, 0)


class SIDWorker(QThread):
    """Background worker executing Spectral Information Divergence classification."""

    progress = Signal(int, str)
    finished = Signal(str, np.ndarray, object, np.ndarray, object, object)
    # args: class_layer_name, class_map, parent_meta, thematic_rgb, rule_images_cube, rule_meta
    failed = Signal(str)

    def __init__(
        self,
        reader: BaseRasterReader,
        parent_meta: RasterMetadata,
        selected_rois: List[ROI],
        max_divergence: Optional[float],
        generate_rules: bool,
        base_name: str,
    ):
        super().__init__()
        self.reader = reader
        self.parent_meta = parent_meta
        self.selected_rois = selected_rois
        self.max_divergence = max_divergence
        self.generate_rules = generate_rules
        self.base_name = base_name

    def run(self):
        try:
            total_bands = self.reader.metadata.bands
            lines = self.reader.metadata.height
            samples = self.reader.metadata.width

            # 1. Read bands into float32 cube (0% - 35%)
            self.progress.emit(5, "Reading raster cube...")
            bands_data = []
            for b in range(total_bands):
                bands_data.append(self.reader.read_band(b))
                pct = 5 + int(30 * (b + 1) / total_bands)
                self.progress.emit(pct, f"Reading band {b + 1}/{total_bands}...")

            cube = np.stack(bands_data, axis=0)  # (bands, lines, samples)
            del bands_data

            # 2. Extract endmember mean spectra from ROIs (35% - 45%)
            self.progress.emit(38, "Extracting endmember spectra from ROIs...")
            ref_list = []
            color_map = {}
            rule_band_details = []

            for idx, roi in enumerate(self.selected_rois):
                spectrum = roi.calculate_mean_spectrum(self.reader)
                if spectrum is None or len(spectrum) != total_bands:
                    raise ValueError(f"Could not compute mean spectrum for ROI: {roi.name}")
                ref_list.append(spectrum)
                color_map[idx] = hex_to_rgb(roi.color)
                rule_band_details.append(
                    BandInfo(
                        index=idx,
                        name=f"SID ({roi.name})",
                        wavelength=None,
                    )
                )

            reference_spectra = np.array(ref_list, dtype=np.float32)

            # 3. Compute SID mapping (45% - 90%)
            self.progress.emit(50, "Computing Spectral Information Divergence...")
            rule_images, class_map = spectral_information_divergence(
                cube=cube,
                reference_spectra=reference_spectra,
                max_divergence=self.max_divergence,
                unclassified_val=-1,
            )
            del cube

            # 4. Generate Thematic RGB visualization (90% - 100%)
            self.progress.emit(92, "Generating thematic color map...")
            thematic_rgb = create_thematic_rgb(
                class_map=class_map,
                palette=color_map,
                unclassified_color=(0, 0, 0),
            )

            # Generate rule images layer if checked
            rule_cube = None
            rule_meta = None
            if self.generate_rules:
                self.progress.emit(96, "Generating divergence rule images...")
                rule_cube = rule_images
                rule_meta = RasterMetadata(
                    width=samples,
                    height=lines,
                    bands=len(self.selected_rois),
                    dtype=np.dtype(np.float32),
                    driver="MEM",
                    crs=self.parent_meta.crs,
                    transform=self.parent_meta.transform,
                    band_details=rule_band_details,
                    nodata_value=None,
                )

            class_layer_name = f"SID Classification ({self.base_name})"
            self.finished.emit(
                class_layer_name,
                class_map,
                self.parent_meta,
                thematic_rgb,
                rule_cube,
                rule_meta,
            )

        except Exception as e:
            self.failed.emit(str(e))


class SIDDialog(QDialog):
    """Dialog for Spectral Information Divergence (SID) classification."""

    result_generated = Signal(str, np.ndarray, object, object)
    # args: layer_name, data_array, parent_meta, optional_rgb_preview

    def __init__(self, layer: RasterLayer, reader: BaseRasterReader, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self._worker: Optional[SIDWorker] = None

        self.setWindowTitle(f"{tr('dialog.sid.title')} - {layer.name}")
        self.resize(780, 640)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Top section: ROI Selection and Preview Graph
        h_split = QHBoxLayout()

        # Left side: ROI Selection Table
        grp_rois = QGroupBox(tr("dialog.sid.grp_rois"))
        v_rois = QVBoxLayout(grp_rois)

        self.table_rois = QTableWidget(0, 3)
        self.table_rois.setHorizontalHeaderLabels([
            tr("dialog.sid.col_select"),
            tr("dialog.sid.col_roi_name"),
            tr("dialog.sid.col_pixels"),
        ])
        self.table_rois.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_rois.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_rois.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_rois.itemChanged.connect(self._on_roi_selection_changed)
        v_rois.addWidget(self.table_rois)

        btn_roi_row = QHBoxLayout()
        self.btn_select_all = QPushButton(tr("dialog.sid.btn_select_all"))
        self.btn_select_all.clicked.connect(self._select_all_rois)
        btn_roi_row.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton(tr("dialog.sid.btn_deselect_all"))
        self.btn_deselect_all.clicked.connect(self._deselect_all_rois)
        btn_roi_row.addWidget(self.btn_deselect_all)
        v_rois.addLayout(btn_roi_row)

        h_split.addWidget(grp_rois, stretch=4)

        # Right side: Spectral Curves Preview
        grp_preview = QGroupBox(tr("dialog.sid.grp_spectral_preview"))
        v_preview = QVBoxLayout(grp_preview)
        self.plot_preview = pg.PlotWidget()
        self.plot_preview.setBackground("#1a1a1a")
        self.plot_preview.showGrid(x=True, y=True, alpha=0.3)
        self.plot_preview.setLabel("left", tr("spectral_profile.axis_y"))
        self.plot_preview.setLabel("bottom", f"{tr('spectral_profile.axis_x_wavelength')} (nm)")
        v_preview.addWidget(self.plot_preview)
        h_split.addWidget(grp_preview, stretch=5)

        main_layout.addLayout(h_split, stretch=1)

        # Parameters section
        grp_params = QGroupBox(tr("dialog.sid.grp_parameters"))
        form_params = QFormLayout(grp_params)

        # Maximum divergence threshold
        self.chk_threshold = QCheckBox(tr("dialog.sid.chk_use_thresh"))
        self.chk_threshold.setChecked(False)
        self.chk_threshold.toggled.connect(self._on_threshold_toggled)

        self.spin_divergence = QDoubleSpinBox()
        self.spin_divergence.setRange(0.0001, 10.0)
        self.spin_divergence.setValue(0.05)
        self.spin_divergence.setDecimals(4)
        self.spin_divergence.setSingleStep(0.005)
        self.spin_divergence.setEnabled(False)

        h_thresh = QHBoxLayout()
        h_thresh.addWidget(self.chk_threshold)
        h_thresh.addWidget(self.spin_divergence)
        form_params.addRow(tr("dialog.sid.lbl_max_divergence"), h_thresh)

        # Rule images checkbox
        self.chk_rule_images = QCheckBox(tr("dialog.sid.chk_rule_images"))
        self.chk_rule_images.setChecked(False)
        form_params.addRow(self.chk_rule_images)

        main_layout.addWidget(grp_params)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Bottom Buttons
        btn_box = QHBoxLayout()
        self.btn_run = QPushButton(tr("dialog.sid.btn_run"))
        self.btn_run.setStyleSheet("background-color: #238636; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_run.clicked.connect(self._run_sid)
        btn_box.addWidget(self.btn_run)

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        main_layout.addLayout(btn_box)

        # Populate ROIs table
        self._populate_rois()

    def _scene_shape(self) -> Tuple[int, int]:
        """Return the (lines, samples) of the target scene."""
        return self.layer.metadata.height, self.layer.metadata.width

    def _populate_rois(self):
        """Fill table with existing layer ROIs."""
        rois = getattr(self.layer, "rois", [])
        self.table_rois.setRowCount(len(rois))
        self.table_rois.blockSignals(True)

        for r, roi in enumerate(rois):
            chk_item = QTableWidgetItem()
            chk_item.setCheckState(Qt.Checked)
            chk_item.setData(Qt.UserRole, r)
            self.table_rois.setItem(r, 0, chk_item)

            name_item = QTableWidgetItem(roi.name)
            name_item.setForeground(QColor(roi.color))
            self.table_rois.setItem(r, 1, name_item)

            pix_count = int(np.sum(roi.get_mask(*self._scene_shape())))
            cnt_item = QTableWidgetItem(str(pix_count))
            self.table_rois.setItem(r, 2, cnt_item)

        self.table_rois.blockSignals(False)
        self._update_curve_preview()

    def _on_threshold_toggled(self, checked: bool):
        self.spin_divergence.setEnabled(checked)

    def _select_all_rois(self):
        self.table_rois.blockSignals(True)
        for r in range(self.table_rois.rowCount()):
            item = self.table_rois.item(r, 0)
            if item:
                item.setCheckState(Qt.Checked)
        self.table_rois.blockSignals(False)
        self._update_curve_preview()

    def _deselect_all_rois(self):
        self.table_rois.blockSignals(True)
        for r in range(self.table_rois.rowCount()):
            item = self.table_rois.item(r, 0)
            if item:
                item.setCheckState(Qt.Unchecked)
        self.table_rois.blockSignals(False)
        self._update_curve_preview()

    def _on_roi_selection_changed(self, item):
        if item and item.column() == 0:
            self._update_curve_preview()

    def _get_selected_rois(self) -> List[ROI]:
        rois = getattr(self.layer, "rois", [])
        selected = []
        for r in range(self.table_rois.rowCount()):
            item = self.table_rois.item(r, 0)
            if item and item.checkState() == Qt.Checked and r < len(rois):
                selected.append(rois[r])
        return selected

    def _update_curve_preview(self):
        """Redraw mean spectral curves for all checked ROIs."""
        self.plot_preview.clear()
        selected = self._get_selected_rois()

        bands = self.layer.metadata.bands
        wls = None
        if self.layer.metadata.band_details:
            w_list = [
                b.wavelength
                for b in self.layer.metadata.band_details
                if b.wavelength is not None
            ]
            if len(w_list) == bands:
                wls = np.array(w_list, dtype=np.float32)

        x_vals = wls if wls is not None else np.arange(1, bands + 1)
        if wls is not None:
            self.plot_preview.setLabel("bottom", f"{tr('spectral_profile.axis_x_wavelength')} (nm)")
        else:
            self.plot_preview.setLabel("bottom", tr("spectral_profile.axis_x_band"))

        for roi in selected:
            spec = roi.calculate_mean_spectrum(self.reader)
            if spec is not None and len(spec) == bands:
                pen = pg.mkPen(color=roi.color, width=2)
                self.plot_preview.plot(x_vals, spec, pen=pen, name=roi.name)

        self.btn_run.setEnabled(len(selected) > 0)

    def _run_sid(self):
        selected_rois = self._get_selected_rois()
        if not selected_rois:
            QMessageBox.warning(
                self,
                tr("dialog.sid.err_no_rois_title"),
                tr("dialog.sid.err_no_rois_msg"),
            )
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        max_div = float(self.spin_divergence.value()) if self.chk_threshold.isChecked() else None

        self._worker = SIDWorker(
            reader=self.reader,
            parent_meta=self.layer.metadata,
            selected_rois=selected_rois,
            max_divergence=max_div,
            generate_rules=self.chk_rule_images.isChecked(),
            base_name=self.layer.name,
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}% - {msg}")

    def _on_worker_finished(
        self,
        class_name: str,
        class_map: np.ndarray,
        parent_meta: object,
        thematic_rgb: np.ndarray,
        rule_cube: Optional[np.ndarray],
        rule_meta: Optional[object],
    ):
        self.progress_bar.setValue(100)
        self.result_generated.emit(class_name, class_map, parent_meta, thematic_rgb)

        if rule_cube is not None and rule_meta is not None:
            rule_name = f"SID Rule Images ({self.layer.name})"
            self.result_generated.emit(rule_name, rule_cube, rule_meta, None)

        event_bus.status_message.emit(
            tr("dialog.sid.status_done").format(name=self.layer.name), 4000
        )
        self.accept()

    def _on_worker_failed(self, err: str):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(
            self,
            tr("dialog.sid.err_compute_title"),
            f"{tr('dialog.sid.err_compute_msg')}\n{err}",
        )
