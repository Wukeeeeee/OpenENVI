"""OpenENVI Support Vector Machine (SVM) Classification Dialog.

Implements supervised SVM classification using training samples extracted from ROIs,
supporting RBF, Linear, Polynomial, and Sigmoid kernels with probability thresholding.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.algorithms.classification import create_thematic_rgb, svm_classification
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


class SVMWorker(QThread):
    """Background worker executing Support Vector Machine (SVM) classification."""

    progress = Signal(int, str)
    finished = Signal(str, np.ndarray, object, np.ndarray, object, object)
    # args: class_name, class_map, parent_meta, thematic_rgb, rule_cube, rule_meta
    failed = Signal(str)

    def __init__(
        self,
        reader: BaseRasterReader,
        parent_meta: RasterMetadata,
        selected_rois: List[ROI],
        kernel: str,
        C: float,
        gamma: Union[str, float],
        degree: int,
        coef0: float,
        probability_threshold: float,
        max_samples_per_class: int,
        generate_rules: bool,
        base_name: str,
    ):
        super().__init__()
        self.reader = reader
        self.parent_meta = parent_meta
        self.selected_rois = selected_rois
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.probability_threshold = probability_threshold
        self.max_samples_per_class = max_samples_per_class
        self.generate_rules = generate_rules
        self.base_name = base_name

    def run(self):
        try:
            total_bands = self.reader.metadata.bands
            lines = self.reader.metadata.height
            samples = self.reader.metadata.width

            # 1. Read bands into float32 cube (0% - 30%)
            self.progress.emit(5, "Reading raster cube...")
            bands_data = []
            for b in range(total_bands):
                bands_data.append(self.reader.read_band(b))
                pct = 5 + int(25 * (b + 1) / total_bands)
                self.progress.emit(pct, f"Reading band {b + 1}/{total_bands}...")

            cube = np.stack(bands_data, axis=0)  # (bands, lines, samples)
            del bands_data

            # 2. Extract training samples from ROIs (30% - 40%)
            self.progress.emit(32, "Extracting training samples from ROIs...")
            training_data: Dict[int, np.ndarray] = {}
            color_map: Dict[int, Tuple[int, int, int]] = {}
            rule_band_details = []

            for idx, roi in enumerate(self.selected_rois):
                # Extract pixel coordinates from ROI mask
                mask = roi.get_mask(lines, samples)
                ys, xs = np.where(mask)
                if len(ys) == 0:
                    continue

                # Sample spectra for all pixels in this ROI: shape (N_pixels, bands)
                roi_pixels = cube[:, ys, xs].T

                # Drop NoData / non-finite spectra so they cannot bias the classifier
                keep = roi._valid_mask(roi_pixels[:, 0], getattr(self.parent_meta, "nodata", None))
                keep &= np.all(np.isfinite(roi_pixels), axis=1)
                roi_pixels = roi_pixels[keep]
                if len(roi_pixels) == 0:
                    continue

                training_data[idx] = roi_pixels

                color_map[idx] = hex_to_rgb(roi.color)
                rule_band_details.append(
                    BandInfo(
                        index=idx,
                        name=f"Prob ({roi.name})",
                        wavelength=None,
                    )
                )

            if len(training_data) < 2:
                raise ValueError("SVM requires at least 2 ROIs with valid pixel samples.")

            def cb(pct, msg):
                self.progress.emit(pct, msg)

            # 3. Fit SVM and Predict (40% - 95%)
            class_map, rule_probs = svm_classification(
                cube=cube,
                training_data=training_data,
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                probability_threshold=self.probability_threshold,
                max_samples_per_class=self.max_samples_per_class,
                progress_callback=cb,
            )
            del cube

            # 4. Generate Thematic RGB visualization
            self.progress.emit(96, "Generating thematic color map...")
            thematic_rgb = create_thematic_rgb(
                class_map=class_map,
                palette=color_map,
                unclassified_color=(0, 0, 0),
            )

            # Rule images metadata
            rule_cube = None
            rule_meta = None
            if self.generate_rules:
                self.progress.emit(98, "Compiling SVM probability rule images...")
                rule_cube = rule_probs
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

            class_layer_name = f"SVM Classification ({self.base_name})"
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


class SVMDialog(QDialog):
    """Dialog for Supervised Support Vector Machine (SVM) classification."""

    result_generated = Signal(str, np.ndarray, object, object)
    # args: layer_name, data_array, parent_meta, optional_rgb_preview

    def __init__(self, layer: RasterLayer, reader: BaseRasterReader, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self._worker: Optional[SVMWorker] = None

        self.setWindowTitle(f"{tr('dialog.svm.title')} - {layer.name}")
        self.resize(780, 680)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Top section: ROI Selection and Preview Graph
        h_split = QHBoxLayout()

        # Left side: ROI Selection Table
        grp_rois = QGroupBox(tr("dialog.svm.grp_rois"))
        v_rois = QVBoxLayout(grp_rois)

        self.table_rois = QTableWidget(0, 3)
        self.table_rois.setHorizontalHeaderLabels([
            tr("dialog.svm.col_select"),
            tr("dialog.svm.col_roi_name"),
            tr("dialog.svm.col_pixels"),
        ])
        self.table_rois.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_rois.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_rois.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_rois.itemChanged.connect(self._on_roi_selection_changed)
        v_rois.addWidget(self.table_rois)

        btn_roi_row = QHBoxLayout()
        self.btn_select_all = QPushButton(tr("dialog.svm.btn_select_all"))
        self.btn_select_all.clicked.connect(self._select_all_rois)
        btn_roi_row.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton(tr("dialog.svm.btn_deselect_all"))
        self.btn_deselect_all.clicked.connect(self._deselect_all_rois)
        btn_roi_row.addWidget(self.btn_deselect_all)
        v_rois.addLayout(btn_roi_row)

        h_split.addWidget(grp_rois, stretch=4)

        # Right side: Spectral Curves Preview
        grp_preview = QGroupBox(tr("dialog.svm.grp_spectral_preview"))
        v_preview = QVBoxLayout(grp_preview)
        self.plot_preview = pg.PlotWidget()
        self.plot_preview.setBackground("#1a1a1a")
        self.plot_preview.showGrid(x=True, y=True, alpha=0.3)
        self.plot_preview.setLabel("left", tr("spectral_profile.axis_y"))
        self.plot_preview.setLabel("bottom", f"{tr('spectral_profile.axis_x_wavelength')} (nm)")
        v_preview.addWidget(self.plot_preview)
        h_split.addWidget(grp_preview, stretch=5)

        main_layout.addLayout(h_split, stretch=1)

        # Middle Section: SVM Parameters
        grp_params = QGroupBox(tr("dialog.svm.grp_parameters"))
        form_params = QFormLayout(grp_params)

        # Kernel selection
        self.cmb_kernel = QComboBox()
        self.cmb_kernel.addItems(["rbf", "linear", "poly", "sigmoid"])
        self.cmb_kernel.currentTextChanged.connect(self._on_kernel_changed)
        form_params.addRow(tr("dialog.svm.lbl_kernel"), self.cmb_kernel)

        # Penalty Parameter C
        self.spin_c = QDoubleSpinBox()
        self.spin_c.setRange(0.01, 100000.0)
        self.spin_c.setValue(100.0)
        self.spin_c.setSingleStep(10.0)
        form_params.addRow(tr("dialog.svm.lbl_c"), self.spin_c)

        # Gamma
        h_gamma = QHBoxLayout()
        self.cmb_gamma_type = QComboBox()
        self.cmb_gamma_type.addItems(["scale", "auto", "custom"])
        self.cmb_gamma_type.currentTextChanged.connect(self._on_gamma_type_changed)
        h_gamma.addWidget(self.cmb_gamma_type)

        self.spin_gamma_val = QDoubleSpinBox()
        self.spin_gamma_val.setRange(0.0001, 1000.0)
        self.spin_gamma_val.setValue(0.01)
        self.spin_gamma_val.setDecimals(4)
        self.spin_gamma_val.setEnabled(False)
        h_gamma.addWidget(self.spin_gamma_val)
        form_params.addRow(tr("dialog.svm.lbl_gamma"), h_gamma)

        # Degree (for Poly)
        self.spin_degree = QSpinBox()
        self.spin_degree.setRange(1, 10)
        self.spin_degree.setValue(3)
        self.spin_degree.setEnabled(False)
        form_params.addRow(tr("dialog.svm.lbl_degree"), self.spin_degree)

        # Coef0 (for Poly / Sigmoid)
        self.spin_coef0 = QDoubleSpinBox()
        self.spin_coef0.setRange(-100.0, 100.0)
        self.spin_coef0.setValue(0.0)
        self.spin_coef0.setEnabled(False)
        form_params.addRow(tr("dialog.svm.lbl_coef0"), self.spin_coef0)

        # Probability Threshold
        self.spin_prob_thresh = QDoubleSpinBox()
        self.spin_prob_thresh.setRange(0.0, 1.0)
        self.spin_prob_thresh.setSingleStep(0.05)
        self.spin_prob_thresh.setValue(0.0)
        self.spin_prob_thresh.setToolTip(tr("dialog.svm.tip_prob_thresh"))
        form_params.addRow(tr("dialog.svm.lbl_prob_thresh"), self.spin_prob_thresh)

        # Training samples per class (0 = use every ROI pixel)
        self.spin_max_samples = QSpinBox()
        self.spin_max_samples.setRange(0, 500000)
        self.spin_max_samples.setValue(2000)
        self.spin_max_samples.setSingleStep(500)
        self.spin_max_samples.setSpecialValueText(tr("dialog.svm.spin_all_samples"))
        self.spin_max_samples.setToolTip(tr("dialog.svm.tip_max_samples"))
        form_params.addRow(tr("dialog.svm.lbl_max_samples"), self.spin_max_samples)

        # Rule images checkbox
        self.chk_rule_images = QCheckBox(tr("dialog.svm.chk_rule_images"))
        self.chk_rule_images.setChecked(False)
        form_params.addRow(self.chk_rule_images)

        main_layout.addWidget(grp_params)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Bottom Buttons
        btn_box = QHBoxLayout()
        self.btn_run = QPushButton(tr("dialog.svm.btn_run"))
        self.btn_run.setStyleSheet("background-color: #238636; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_run.clicked.connect(self._run_svm)
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

    def _on_kernel_changed(self, kernel: str):
        """Update active parameter inputs based on chosen kernel."""
        is_poly = (kernel == "poly")
        is_sig = (kernel == "sigmoid")
        is_linear = (kernel == "linear")

        self.spin_degree.setEnabled(is_poly)
        self.spin_coef0.setEnabled(is_poly or is_sig)
        self.cmb_gamma_type.setEnabled(not is_linear)
        self.spin_gamma_val.setEnabled(not is_linear and self.cmb_gamma_type.currentText() == "custom")

    def _on_gamma_type_changed(self, gamma_type: str):
        self.spin_gamma_val.setEnabled(gamma_type == "custom")

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

        self.btn_run.setEnabled(len(selected) >= 2)

    def _run_svm(self):
        selected_rois = self._get_selected_rois()
        if len(selected_rois) < 2:
            QMessageBox.warning(
                self,
                tr("dialog.svm.err_min_rois_title"),
                tr("dialog.svm.err_min_rois_msg"),
            )
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        gamma_type = self.cmb_gamma_type.currentText()
        gamma_val: Union[str, float] = (
            float(self.spin_gamma_val.value()) if gamma_type == "custom" else gamma_type
        )

        self._worker = SVMWorker(
            reader=self.reader,
            parent_meta=self.layer.metadata,
            selected_rois=selected_rois,
            kernel=self.cmb_kernel.currentText(),
            C=float(self.spin_c.value()),
            gamma=gamma_val,
            degree=int(self.spin_degree.value()),
            coef0=float(self.spin_coef0.value()),
            probability_threshold=float(self.spin_prob_thresh.value()),
            max_samples_per_class=int(self.spin_max_samples.value()),
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
            rule_name = f"SVM Rule Probabilities ({self.layer.name})"
            self.result_generated.emit(rule_name, rule_cube, rule_meta, None)

        event_bus.status_message.emit(
            tr("dialog.svm.status_done").format(name=self.layer.name), 4000
        )
        self.accept()

    def _on_worker_failed(self, err: str):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(
            self,
            tr("dialog.svm.err_compute_title"),
            f"{tr('dialog.svm.err_compute_msg')}\n{err}",
        )
