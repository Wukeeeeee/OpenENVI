"""OpenENVI Spectral Angle Mapper (SAM) Dialog.

Implements supervised Spectral Angle Mapper classification using reference
endmember spectra derived from Regions of Interest (ROIs).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
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
from core.algorithms.spectral import spectral_angle_mapper
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


class SAMWorker(QThread):
    """Background worker executing Spectral Angle Mapper classification."""

    progress = Signal(int, str)
    finished = Signal(str, np.ndarray, object, np.ndarray, object, object)
    # args: class_layer_name, class_map, parent_meta, thematic_rgb, rule_images_cube, rule_meta
    failed = Signal(str)

    def __init__(
        self,
        reader: BaseRasterReader,
        parent_meta: RasterMetadata,
        selected_rois: List[ROI],
        max_angle: float,
        generate_rules: bool,
        base_name: str,
    ):
        super().__init__()
        self.reader = reader
        self.parent_meta = parent_meta
        self.selected_rois = selected_rois
        self.max_angle = max_angle
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
                if spectrum is None:
                    # Fallback to zeros if ROI had no pixels
                    spectrum = np.zeros(total_bands, dtype=np.float32)
                ref_list.append(spectrum)
                color_map[idx] = hex_to_rgb(roi.color)
                rule_band_details.append(
                    BandInfo(
                        index=idx,
                        name=f"Rule: {roi.name}",
                        wavelength_unit="rad",
                    )
                )

            reference_spectra = np.vstack(ref_list)  # (M, bands)

            # 3. Compute SAM spectral angles (45% - 80%)
            self.progress.emit(50, "Executing Spectral Angle Mapper...")
            rule_images, class_map = spectral_angle_mapper(
                cube,
                reference_spectra,
                max_angle=self.max_angle,
                unclassified_val=-1,
            )
            del cube

            # 4. Generate Thematic RGB Map (80% - 95%)
            self.progress.emit(85, "Generating thematic classification display...")
            thematic_rgb = create_thematic_rgb(
                class_map,
                palette=color_map,
                unclassified_color=(0, 0, 0),
            )

            # Prepare Rule Images metadata if requested
            rule_cube = None
            rule_meta = None
            if self.generate_rules:
                rule_cube = rule_images  # (num_endmembers, lines, samples)
                rule_meta = RasterMetadata(
                    width=samples,
                    height=lines,
                    bands=len(self.selected_rois),
                    dtype="float32",
                    crs=self.parent_meta.crs,
                    transform=self.parent_meta.transform,
                    interleave="BSQ",
                    band_details=rule_band_details,
                    raw_header=dict(self.parent_meta.raw_header) if self.parent_meta.raw_header else {},
                )

            class_layer_name = f"SAM Classify ({self.base_name}, max={self.max_angle:.2f} rad)"

            self.progress.emit(100, "SAM Classification Complete!")
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


class SAMDialog(QDialog):
    """Dialog for configuring and executing Spectral Angle Mapper (SAM) classification."""

    result_generated = Signal(str, np.ndarray, object, object)

    def __init__(
        self,
        layer: RasterLayer,
        reader: BaseRasterReader,
        roi_manager=None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self.roi_manager = roi_manager
        self._worker: Optional[SAMWorker] = None

        self.setWindowTitle(f"{tr('dialog.sam.title')} - {layer.name}")
        self.resize(680, 560)

        self._init_ui()
        self._populate_rois()
        self._update_curve_preview()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Info & Layer Banner
        lbl_info = QLabel(
            f"<b>{tr('dialog.sam.layer')}:</b> {self.layer.name} "
            f"({self.layer.metadata.bands} {tr('dialog.sam.bands')}, "
            f"{self.layer.metadata.width}x{self.layer.metadata.height})"
        )
        layout.addWidget(lbl_info)

        # 2. Endmember Selection Group
        grp_endmembers = QGroupBox(tr("dialog.sam.grp_endmembers"))
        l_em = QVBoxLayout(grp_endmembers)

        self.table_rois = QTableWidget()
        self.table_rois.setColumnCount(4)
        self.table_rois.setHorizontalHeaderLabels([
            tr("dialog.sam.col_select"),
            tr("dialog.sam.col_color"),
            tr("dialog.sam.col_name"),
            tr("dialog.sam.col_pixels"),
        ])
        self.table_rois.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_rois.itemChanged.connect(self._on_roi_selection_changed)
        l_em.addWidget(self.table_rois)

        btn_row = QHBoxLayout()
        self.btn_select_all = QPushButton(tr("dialog.sam.btn_select_all"))
        self.btn_select_all.clicked.connect(self._select_all_rois)
        btn_row.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton(tr("dialog.sam.btn_deselect_all"))
        self.btn_deselect_all.clicked.connect(self._deselect_all_rois)
        btn_row.addWidget(self.btn_deselect_all)

        btn_row.addStretch()
        l_em.addLayout(btn_row)

        layout.addWidget(grp_endmembers, stretch=2)

        # 3. Spectral Curves Preview
        grp_plot = QGroupBox(tr("dialog.sam.grp_spectrum_preview"))
        l_plot = QVBoxLayout(grp_plot)
        self.plot_preview = pg.PlotWidget()
        self.plot_preview.setBackground("#1e1e1e")
        self.plot_preview.showGrid(x=True, y=True, alpha=0.3)
        self.plot_preview.setLabel("left", tr("dialog.sam.lbl_reflectance"))
        self.plot_preview.setLabel("bottom", tr("dialog.sam.lbl_bands"))
        l_plot.addWidget(self.plot_preview)
        layout.addWidget(grp_plot, stretch=2)

        # 4. Parameters
        grp_params = QGroupBox(tr("dialog.sam.grp_params"))
        l_params = QVBoxLayout(grp_params)

        h_angle = QHBoxLayout()
        h_angle.addWidget(QLabel(tr("dialog.sam.lbl_max_angle")))
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(0.001, 3.1416)
        self.spin_angle.setDecimals(3)
        self.spin_angle.setSingleStep(0.01)
        self.spin_angle.setValue(0.100)
        self.spin_angle.valueChanged.connect(self._on_angle_changed)
        h_angle.addWidget(self.spin_angle)

        self.lbl_angle_deg = QLabel("(5.73°)")
        self.lbl_angle_deg.setStyleSheet("color: #8b949e;")
        h_angle.addWidget(self.lbl_angle_deg)
        h_angle.addStretch()
        l_params.addLayout(h_angle)

        self.chk_rule_images = QCheckBox(tr("dialog.sam.chk_rule_images"))
        self.chk_rule_images.setChecked(True)
        l_params.addWidget(self.chk_rule_images)

        layout.addWidget(grp_params)

        # 5. Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # 6. Button Box
        btn_box = QHBoxLayout()
        self.btn_run = QPushButton(tr("dialog.sam.btn_run"))
        self.btn_run.setStyleSheet("background-color: #238636; color: white; font-weight: bold; padding: 6px;")
        self.btn_run.clicked.connect(self._run_sam)
        btn_box.addWidget(self.btn_run)

        self.btn_close = QPushButton(tr("dialog.btn_cancel"))
        self.btn_close.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_close)

        layout.addLayout(btn_box)

    def _populate_rois(self):
        """Populate ROIs available on this layer."""
        self.table_rois.blockSignals(True)
        rois = getattr(self.layer, "rois", [])
        self.table_rois.setRowCount(len(rois))

        h = self.layer.metadata.height
        w = self.layer.metadata.width

        for row, roi in enumerate(rois):
            # Checkbox
            item_check = QTableWidgetItem()
            item_check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item_check.setCheckState(Qt.Checked)
            self.table_rois.setItem(row, 0, item_check)

            # Color badge
            item_color = QTableWidgetItem()
            item_color.setFlags(Qt.ItemIsEnabled)
            item_color.setBackground(QColor(roi.color))
            self.table_rois.setItem(row, 1, item_color)

            # Name
            item_name = QTableWidgetItem(roi.name)
            item_name.setFlags(Qt.ItemIsEnabled)
            self.table_rois.setItem(row, 2, item_name)

            # Pixel count
            mask = roi.get_mask(h, w)
            pix_count = int(np.sum(mask))
            item_pixels = QTableWidgetItem(f"{pix_count:,}")
            item_pixels.setFlags(Qt.ItemIsEnabled)
            self.table_rois.setItem(row, 3, item_pixels)

        self.table_rois.blockSignals(False)

        if not rois:
            self.btn_run.setEnabled(False)
            QMessageBox.information(
                self,
                tr("dialog.sam.no_rois_title"),
                tr("dialog.sam.no_rois_msg"),
            )

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

    def _on_angle_changed(self, val: float):
        deg = np.degrees(val)
        self.lbl_angle_deg.setText(f"({deg:.2f}°)")

    def _run_sam(self):
        selected_rois = self._get_selected_rois()
        if not selected_rois:
            QMessageBox.warning(
                self,
                tr("dialog.sam.err_no_rois_title"),
                tr("dialog.sam.err_no_rois_msg"),
            )
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self._worker = SAMWorker(
            reader=self.reader,
            parent_meta=self.layer.metadata,
            selected_rois=selected_rois,
            max_angle=float(self.spin_angle.value()),
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
        # Emit classification layer with integer class_map and thematic RGB rendering
        self.result_generated.emit(class_name, class_map, parent_meta, thematic_rgb)

        # Emit rule images if generated
        if rule_cube is not None and rule_meta is not None:
            rule_name = f"SAM Rule Images ({self.layer.name})"
            self.result_generated.emit(rule_name, rule_cube, rule_meta, None)

        event_bus.status_message.emit(
            f"SAM classification completed for {self.layer.name}", 4000
        )
        self.accept()

    def _on_worker_failed(self, err: str):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(
            self,
            tr("dialog.sam.err_compute_title"),
            f"{tr('dialog.sam.err_compute_msg')}\n{err}",
        )
