"""OpenENVI Image Classification Dialog.

Supports unsupervised classification (K-Means & ISODATA) with thematic
color rendering and class cluster center reporting.
"""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from core.algorithms.classification import (
    create_thematic_rgb,
    isodata_clustering,
    kmeans_clustering,
)
from core.i18n import tr
from core.models import RasterLayer


class ClassificationDialog(QDialog):
    """Dialog for unsupervised remote sensing classification."""

    result_generated = Signal(str, np.ndarray)  # layer_name, 3D thematic RGB array

    def __init__(self, layer: RasterLayer, reader, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self.setWindowTitle(tr("dialog.classification.title"))
        self.resize(420, 260)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()

        # Method
        self.cb_method = QComboBox()
        self.cb_method.addItems([
            tr("classification.method_kmeans"),
            tr("classification.method_isodata"),
        ])
        form.addRow(tr("dialog.classification.method"), self.cb_method)

        # Classes
        self.spin_classes = QSpinBox()
        self.spin_classes.setRange(2, 10)
        self.spin_classes.setValue(4)
        form.addRow(tr("dialog.classification.classes"), self.spin_classes)

        # Iterations
        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(5, 50)
        self.spin_iters.setValue(15)
        form.addRow(tr("dialog.classification.max_iter"), self.spin_iters)

        layout.addLayout(form)
        layout.addStretch()

        # Buttons
        btn_box = QHBoxLayout()
        self.btn_run = QPushButton(tr("dialog.classification.btn_run"))
        self.btn_run.setStyleSheet("background-color: #238636; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self._run_classification)
        btn_box.addWidget(self.btn_run)

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        layout.addLayout(btn_box)

    def _run_classification(self) -> None:
        """Run classification and emit thematic RGB map."""
        method_idx = self.cb_method.currentIndex()
        num_classes = self.spin_classes.value()
        max_iters = self.spin_iters.value()

        try:
            # Read multi-band data
            bands_data = [self.reader.read_band(b) for b in range(self.layer.metadata.bands)]
            cube = np.stack(bands_data, axis=0)  # (bands, lines, samples)

            if method_idx == 0:
                class_map, centers = kmeans_clustering(cube, num_classes=num_classes, max_iter=max_iters)
                method_name = "K-Means"
            else:
                class_map, centers = isodata_clustering(cube, initial_classes=num_classes, max_iter=max_iters)
                method_name = "ISODATA"

            thematic_rgb = create_thematic_rgb(class_map)
            layer_name = f"Classify ({method_name}, K={num_classes})"
            self.result_generated.emit(layer_name, thematic_rgb)
            self.accept()
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("classification.err_title"),
                f"{tr('classification.err_msg')}\n{e}",
            )
