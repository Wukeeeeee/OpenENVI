"""OpenENVI Confusion Matrix & Classification Accuracy Assessment Dialog.

Provides interactive comparison between a classified thematic map
and ground truth reference data, displaying confusion matrix, Overall Accuracy,
and Kappa coefficient.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.algorithms.accuracy import compute_confusion_matrix
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import RasterLayer


class AccuracyAssessmentDialog(QDialog):
    """Dialog for computing and displaying confusion matrix and accuracy metrics."""

    def __init__(
        self,
        layers: Dict[str, Tuple[RasterLayer, BaseRasterReader]],
        active_layer_id: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.layers = layers
        self.active_layer_id = active_layer_id
        self._accuracy_result = None

        self.setWindowTitle(tr("toolbox.tool_accuracy"))
        self.resize(720, 560)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Layer Selection Group
        grp_layers = QGroupBox(tr("accuracy.grp_layers"))
        l_inputs = QVBoxLayout(grp_layers)

        # Classified layer
        h_cls = QHBoxLayout()
        h_cls.addWidget(QLabel(tr("accuracy.classified")))
        self.cb_classified = QComboBox()
        h_cls.addWidget(self.cb_classified, stretch=1)
        l_inputs.addLayout(h_cls)

        # Reference layer
        h_ref = QHBoxLayout()
        h_ref.addWidget(QLabel(tr("accuracy.reference")))
        self.cb_reference = QComboBox()
        h_ref.addWidget(self.cb_reference, stretch=1)
        l_inputs.addLayout(h_ref)

        # Ignore unclassified pixels (-1)
        self.chk_ignore_unclassified = QCheckBox(tr("accuracy.ignore_unclassified"))
        self.chk_ignore_unclassified.setChecked(True)
        l_inputs.addWidget(self.chk_ignore_unclassified)

        self._populate_layers()
        layout.addWidget(grp_layers)

        # Compute button
        self.btn_compute = QPushButton(tr("accuracy.btn_compute"))
        self.btn_compute.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 6px;")
        self.btn_compute.clicked.connect(self._compute_accuracy)
        layout.addWidget(self.btn_compute)

        # 2. Summary stats labels
        grp_summary = QGroupBox(tr("accuracy.grp_metrics"))
        l_summ = QHBoxLayout(grp_summary)
        self.lbl_oa = QLabel(tr("accuracy.oa_idle"))
        self.lbl_oa.setStyleSheet("font-size: 14px; font-weight: bold; color: #2ecc71;")
        l_summ.addWidget(self.lbl_oa)

        self.lbl_kappa = QLabel(tr("accuracy.kappa_idle"))
        self.lbl_kappa.setStyleSheet("font-size: 14px; font-weight: bold; color: #f1c40f;")
        l_summ.addWidget(self.lbl_kappa)
        layout.addWidget(grp_summary)

        # 3. Matrix Table
        self.table = QTableWidget()
        layout.addWidget(self.table, stretch=1)

        # Action Buttons
        btn_bar = QHBoxLayout()
        self.btn_export = QPushButton(tr("accuracy.btn_export"))
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_report)
        btn_bar.addWidget(self.btn_export)

        btn_bar.addStretch()

        self.btn_close = QPushButton(tr("dialog.btn_ok"))
        self.btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(self.btn_close)

        layout.addLayout(btn_bar)

    def _populate_layers(self):
        for lid, (layer, _) in self.layers.items():
            self.cb_classified.addItem(layer.name, lid)
            self.cb_reference.addItem(layer.name, lid)

        if self.active_layer_id and self.active_layer_id in self.layers:
            idx = self.cb_classified.findData(self.active_layer_id)
            if idx >= 0:
                self.cb_classified.setCurrentIndex(idx)
            # Default reference to another layer if possible
            if self.cb_reference.count() > 1:
                ref_idx = 1 if idx == 0 else 0
                self.cb_reference.setCurrentIndex(ref_idx)

    def _compute_accuracy(self):
        cls_id = self.cb_classified.currentData()
        ref_id = self.cb_reference.currentData()

        if not cls_id or not ref_id:
            QMessageBox.warning(
                self,
                tr("accuracy.err_selection_title"),
                tr("accuracy.err_selection_msg"),
            )
            return

        cls_layer, cls_reader = self.layers[cls_id]
        ref_layer, ref_reader = self.layers[ref_id]

        try:
            # Read first band of each
            c_data = cls_reader.read_band(0)
            r_data = ref_reader.read_band(0)

            if c_data.shape != r_data.shape:
                from core.algorithms.pansharpen import resample_band_to_grid
                r_data = resample_band_to_grid(r_data, c_data.shape[0], c_data.shape[1])

            nodata_val = -1 if self.chk_ignore_unclassified.isChecked() else (
                cls_layer.metadata.nodata if cls_layer.metadata.nodata is not None else None
            )

            res = compute_confusion_matrix(c_data, r_data, nodata_val=nodata_val)
            self._accuracy_result = res

            self.lbl_oa.setText(f"{tr('accuracy.oa')} {res['overall_accuracy']:.2f}%")
            self.lbl_kappa.setText(f"{tr('accuracy.kappa')} {res['kappa']:.4f}")
            self.btn_export.setEnabled(True)

            self._display_matrix(res)
        except Exception as e:
            QMessageBox.critical(self, tr("dialog.error"), f"{tr('accuracy.err_compute')}: {e}")

    def _display_matrix(self, res: Dict):
        matrix = res["matrix"]
        classes = res["classes"]
        k = len(classes)

        # Columns: Class, C_0, C_1, ..., Total, PA (%)
        col_headers = [tr("accuracy.col_ref_pred")] + [f"C_{c}" for c in classes] + [tr("accuracy.col_total"), tr("accuracy.col_pa")]
        self.table.setColumnCount(len(col_headers))
        self.table.setHorizontalHeaderLabels(col_headers)
        self.table.setRowCount(k + 1)  # +1 for UA row

        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)
        pa = res["producers_accuracy"]
        ua = res["users_accuracy"]

        for r_idx, cls_val in enumerate(classes):
            self.table.setItem(r_idx, 0, QTableWidgetItem(f"{tr('accuracy.class_prefix')} {cls_val}"))
            for c_idx in range(k):
                val = matrix[r_idx, c_idx]
                item = QTableWidgetItem(str(val))
                if r_idx == c_idx:
                    item.setBackground(QColor("#1e4620"))  # Highlight correct diagonal
                self.table.setItem(r_idx, c_idx + 1, item)

            self.table.setItem(r_idx, k + 1, QTableWidgetItem(str(row_sums[r_idx])))
            self.table.setItem(r_idx, k + 2, QTableWidgetItem(f"{pa[cls_val]:.2f}%"))

        # User's accuracy row
        ua_row = k
        self.table.setItem(ua_row, 0, QTableWidgetItem(tr("accuracy.col_ua")))
        for c_idx, cls_val in enumerate(classes):
            self.table.setItem(ua_row, c_idx + 1, QTableWidgetItem(f"{ua[cls_val]:.2f}%"))
        self.table.setItem(ua_row, k + 1, QTableWidgetItem(str(np.sum(matrix))))
        self.table.setItem(ua_row, k + 2, QTableWidgetItem(f"{res['overall_accuracy']:.2f}%"))

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def _export_report(self):
        if not self._accuracy_result:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            tr("accuracy.save_title"),
            "accuracy_report.txt",
            "Text Files (*.txt);;CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self._accuracy_result["report"])
            QMessageBox.information(
                self,
                tr("accuracy.export_success_title"),
                f"{tr('accuracy.export_success_msg')}\n{file_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, tr("dialog.error"), f"{tr('accuracy.export_err_msg')}: {e}")
