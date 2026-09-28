"""OpenENVI Maximum Likelihood Classification (MLC) Dialog.

Implements supervised Maximum Likelihood Classification using training samples
delineated from Regions of Interest (ROIs). Supports Chi-Square probability thresholding
to filter unclassified pixels, sample-proportional priors, and optional Mahalanobis
distance rule images.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
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

from core.algorithms.classification import create_thematic_rgb, maximum_likelihood_classification
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


class MLCWorker(QThread):
    """Background worker executing supervised Maximum Likelihood Classification."""

    progress = Signal(int, str)
    finished = Signal(str, np.ndarray, object, np.ndarray, object, object)
    # args: class_layer_name, class_map, parent_meta, thematic_rgb, rule_cube, rule_meta
    failed = Signal(str)

    def __init__(
        self,
        reader: BaseRasterReader,
        parent_meta: RasterMetadata,
        selected_rois: List[ROI],
        probability_threshold: float,
        use_sample_priors: bool,
        generate_rules: bool,
        base_name: str,
        parent=None,
    ):
        super().__init__(parent)
        self.reader = reader
        self.parent_meta = parent_meta
        self.selected_rois = selected_rois
        self.probability_threshold = probability_threshold
        self.use_sample_priors = use_sample_priors
        self.generate_rules = generate_rules
        self.base_name = base_name

    def run(self):
        try:
            total_bands = self.reader.metadata.bands
            lines = self.reader.metadata.height
            samples = self.reader.metadata.width

            # 1. Read bands into float32 cube (0% - 30%)
            self.progress.emit(5, "Reading raster bands...")
            bands_data = []
            for b in range(total_bands):
                bands_data.append(self.reader.read_band(b))
                pct = 5 + int(25 * (b + 1) / total_bands)
                self.progress.emit(pct, f"Reading band {b + 1}/{total_bands}...")

            cube = np.stack(bands_data, axis=0)  # (bands, lines, samples)
            del bands_data

            # 2. Extract training pixels for each ROI (30% - 40%)
            self.progress.emit(32, "Extracting training samples from ROIs...")
            training_data: Dict[int, np.ndarray] = {}
            color_map: Dict[int, Tuple[int, int, int]] = {}
            rule_band_details: List[BandInfo] = []

            for idx, roi in enumerate(self.selected_rois):
                mask = roi.get_mask(lines, samples)
                roi_pixels = cube[:, mask].T  # (N_pixels, bands)
                finite_mask = np.all(np.isfinite(roi_pixels), axis=1)
                valid_samples = roi_pixels[finite_mask]

                if len(valid_samples) == 0:
                    raise ValueError(f"ROI '{roi.name}' contains 0 valid training pixels.")

                training_data[idx] = valid_samples
                color_map[idx] = hex_to_rgb(roi.color)
                rule_band_details.append(
                    BandInfo(
                        index=idx,
                        name=f"Mahalanobis Dist: {roi.name}",
                        wavelength_unit="dist",
                    )
                )

            # 3. Execute Maximum Likelihood Classification (40% - 90%)
            self.progress.emit(40, "Computing class statistics and Mahalanobis distances...")
            class_map, dists = maximum_likelihood_classification(
                cube=cube,
                training_data=training_data,
                probability_threshold=self.probability_threshold,
                use_sample_priors=self.use_sample_priors,
                progress_callback=lambda p, msg: self.progress.emit(p, msg),
            )
            del cube

            # 4. Generate Thematic RGB Map (90% - 96%)
            self.progress.emit(92, "Rendering thematic classification map...")
            thematic_rgb = create_thematic_rgb(
                class_map=class_map,
                palette=color_map,
                unclassified_color=(0, 0, 0),
            )

            # 5. Build Rule Images if requested
            rule_cube = None
            rule_meta = None
            if self.generate_rules:
                rule_cube = dists  # (num_classes, lines, samples)
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

            thresh_str = f"thresh={self.probability_threshold:.3f}" if self.probability_threshold > 0 else "no thresh"
            class_layer_name = f"MLC Classify ({self.base_name}, {thresh_str})"

            self.progress.emit(100, "Maximum Likelihood Classification Complete!")
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


class MaximumLikelihoodDialog(QDialog):
    """Dialog for configuring and executing Supervised Maximum Likelihood Classification."""

    result_generated = Signal(str, np.ndarray, object, object)
    open_roi_tool_requested = Signal(str)

    def __init__(
        self,
        layer: RasterLayer,
        reader: BaseRasterReader,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self._worker: Optional[MLCWorker] = None

        self.setWindowTitle(f"{tr('dialog.maxlik.title')} - {layer.name}")
        self.resize(600, 520)

        self._init_ui()
        self._populate_rois()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Target Raster Info
        lbl_info = QLabel(
            f"<b>{tr('dialog.maxlik.layer')}:</b> {self.layer.name} "
            f"({self.layer.metadata.bands} {tr('dialog.maxlik.bands')}, "
            f"{self.layer.metadata.width}x{self.layer.metadata.height})"
        )
        layout.addWidget(lbl_info)

        # 2. ROI Training Classes Group
        grp_classes = QGroupBox(tr("dialog.maxlik.grp_classes"))
        l_cls = QVBoxLayout(grp_classes)

        self.table_rois = QTableWidget()
        self.table_rois.setColumnCount(4)
        self.table_rois.setHorizontalHeaderLabels([
            tr("dialog.maxlik.col_select"),
            tr("dialog.maxlik.col_color"),
            tr("dialog.maxlik.col_name"),
            tr("dialog.maxlik.col_pixels"),
        ])
        self.table_rois.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        l_cls.addWidget(self.table_rois)

        btn_row = QHBoxLayout()
        self.btn_select_all = QPushButton(tr("dialog.maxlik.btn_select_all"))
        self.btn_select_all.clicked.connect(self._select_all_rois)
        btn_row.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton(tr("dialog.maxlik.btn_deselect_all"))
        self.btn_deselect_all.clicked.connect(self._deselect_all_rois)
        btn_row.addWidget(self.btn_deselect_all)

        self.btn_open_roi = QPushButton(tr("dialog.maxlik.btn_open_roi"))
        self.btn_open_roi.setStyleSheet("background-color: #3b4252; color: #eceff4;")
        self.btn_open_roi.clicked.connect(self._on_open_roi_tool)
        btn_row.addWidget(self.btn_open_roi)

        btn_row.addStretch()
        l_cls.addLayout(btn_row)
        layout.addWidget(grp_classes)

        # 3. Parameters Group
        grp_params = QGroupBox(tr("dialog.maxlik.grp_params"))
        l_params = QVBoxLayout(grp_params)

        row_thresh = QHBoxLayout()
        lbl_thresh = QLabel(tr("dialog.maxlik.lbl_threshold"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 0.50)
        self.spin_threshold.setSingleStep(0.01)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setValue(0.0)
        self.spin_threshold.setToolTip(tr("dialog.maxlik.tip_threshold"))
        row_thresh.addWidget(lbl_thresh)
        row_thresh.addWidget(self.spin_threshold)
        row_thresh.addStretch()
        l_params.addLayout(row_thresh)

        self.chk_priors = QCheckBox(tr("dialog.maxlik.chk_sample_priors"))
        self.chk_priors.setChecked(False)
        l_params.addWidget(self.chk_priors)

        self.chk_rule_images = QCheckBox(tr("dialog.maxlik.chk_rule_images"))
        self.chk_rule_images.setChecked(False)
        l_params.addWidget(self.chk_rule_images)

        layout.addWidget(grp_params)

        # 4. Progress bar & status label (hidden until execution starts)
        self.lbl_progress = QLabel("")
        self.lbl_progress.setStyleSheet("color: #8e9297; font-size: 11px;")
        self.lbl_progress.setVisible(False)
        layout.addWidget(self.lbl_progress)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #4e535a; border-radius: 3px; text-align: center; height: 16px; background-color: #1e1f22; color: #ffffff; }"
            "QProgressBar::chunk { background-color: #238636; border-radius: 2px; }"
        )
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        # 5. Dialog Actions
        btn_box = QHBoxLayout()
        self.btn_run = QPushButton(tr("dialog.maxlik.btn_run"))
        self.btn_run.setStyleSheet("background-color: #238636; color: white; font-weight: bold; min-height: 28px; padding: 0 16px;")
        self.btn_run.clicked.connect(self._run_classification)
        btn_box.addWidget(self.btn_run)

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_box.addWidget(self.btn_cancel)

        layout.addLayout(btn_box)

    def _populate_rois(self):
        """Populate the ROI table from active layer's ROIs."""
        rois: List[ROI] = getattr(self.layer, "rois", [])
        self.table_rois.setRowCount(len(rois))

        h, w = self.layer.metadata.height, self.layer.metadata.width

        for row, roi in enumerate(rois):
            # Column 0: Checkbox
            item_check = QTableWidgetItem()
            item_check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item_check.setCheckState(Qt.Checked)
            item_check.setData(Qt.UserRole, roi)
            self.table_rois.setItem(row, 0, item_check)

            # Column 1: Color indicator
            item_color = QTableWidgetItem("   ")
            item_color.setFlags(Qt.ItemIsEnabled)
            item_color.setBackground(QColor(roi.color))
            self.table_rois.setItem(row, 1, item_color)

            # Column 2: Name
            item_name = QTableWidgetItem(roi.name)
            item_name.setFlags(Qt.ItemIsEnabled)
            self.table_rois.setItem(row, 2, item_name)

            # Column 3: Pixel count
            mask = roi.get_mask(h, w)
            pix_count = int(np.sum(mask))
            item_pix = QTableWidgetItem(f"{pix_count:,}")
            item_pix.setFlags(Qt.ItemIsEnabled)
            item_pix.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table_rois.setItem(row, 3, item_pix)

        if not rois:
            self.btn_run.setEnabled(False)

    def _select_all_rois(self):
        for r in range(self.table_rois.rowCount()):
            item = self.table_rois.item(r, 0)
            if item:
                item.setCheckState(Qt.Checked)

    def _deselect_all_rois(self):
        for r in range(self.table_rois.rowCount()):
            item = self.table_rois.item(r, 0)
            if item:
                item.setCheckState(Qt.Unchecked)

    def _on_open_roi_tool(self):
        self.open_roi_tool_requested.emit(self.layer.layer_id)
        self.reject()

    def _get_selected_rois(self) -> List[ROI]:
        selected = []
        for r in range(self.table_rois.rowCount()):
            item = self.table_rois.item(r, 0)
            if item and item.checkState() == Qt.Checked:
                roi = item.data(Qt.UserRole)
                if roi is not None:
                    selected.append(roi)
        return selected

    def _run_classification(self):
        selected_rois = self._get_selected_rois()
        h, w = self.layer.metadata.height, self.layer.metadata.width

        # Filter ROIs that actually contain pixels
        valid_rois = [roi for roi in selected_rois if np.sum(roi.get_mask(h, w)) > 0]

        if len(valid_rois) < 2:
            QMessageBox.warning(
                self,
                tr("dialog.maxlik.no_rois_title"),
                tr("dialog.maxlik.err_min_classes"),
            )
            return

        self.btn_run.setEnabled(False)
        self.btn_select_all.setEnabled(False)
        self.btn_deselect_all.setEnabled(False)
        self.spin_threshold.setEnabled(False)
        self.chk_priors.setEnabled(False)
        self.chk_rule_images.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress.setVisible(True)
        self.lbl_progress.setText("Starting Maximum Likelihood Classification...")

        self._worker = MLCWorker(
            reader=self.reader,
            parent_meta=self.layer.metadata,
            selected_rois=valid_rois,
            probability_threshold=self.spin_threshold.value(),
            use_sample_priors=self.chk_priors.isChecked(),
            generate_rules=self.chk_rule_images.isChecked(),
            base_name=self.layer.name,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.lbl_progress.setText(message)

    @Slot(str, np.ndarray, object, np.ndarray, object, object)
    def _on_finished(
        self,
        layer_name: str,
        class_map: np.ndarray,
        meta: object,
        thematic_rgb: np.ndarray,
        rule_cube: Optional[np.ndarray],
        rule_meta: Optional[object],
    ):
        event_bus.status_message.emit(f"MLC completed: {layer_name}", 4000)
        self.result_generated.emit(layer_name, class_map, meta, thematic_rgb)

        if rule_cube is not None and rule_meta is not None:
            rule_name = f"MLC Rules ({self.layer.name})"
            self.result_generated.emit(rule_name, rule_cube, rule_meta, None)

        self.accept()

    @Slot(str)
    def _on_failed(self, error_msg: str):
        self.btn_run.setEnabled(True)
        self.btn_select_all.setEnabled(True)
        self.btn_deselect_all.setEnabled(True)
        self.spin_threshold.setEnabled(True)
        self.chk_priors.setEnabled(True)
        self.chk_rule_images.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)
        QMessageBox.critical(
            self,
            tr("dialog.maxlik.err_title"),
            f"{error_msg}",
        )

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
        self.reject()

    def closeEvent(self, event):
        self._on_cancel()
        super().closeEvent(event)
