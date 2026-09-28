"""OpenENVI Image Classification Dialog.

Supports unsupervised classification (K-Means & ISODATA) with thematic
color rendering and background worker threading with real-time progress feedback.
"""

from typing import Optional
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from core.algorithms.classification import (
    create_thematic_rgb,
    isodata_clustering,
    kmeans_clustering,
)
from core.events import event_bus
from core.i18n import tr
from core.models import RasterLayer


class ClassificationWorker(QThread):
    """Background worker thread executing classification without GUI blocking."""

    progress = Signal(int, str)  # percent, status_text
    finished = Signal(str, np.ndarray, object, object)  # layer_name, 2D class_map, meta, 3D thematic RGB array
    failed = Signal(str)  # error_message

    def __init__(
        self,
        reader,
        total_bands: int,
        method_idx: int,
        num_classes: int,
        max_iters: int,
        parent=None,
    ):
        super().__init__(parent)
        self.reader = reader
        self.total_bands = total_bands
        self.method_idx = method_idx
        self.num_classes = num_classes
        self.max_iters = max_iters

    def run(self) -> None:
        try:
            # 1. Read bands sequentially with progress updates (0% - 30%)
            self.progress.emit(5, tr("classification.status_reading_bands") if tr("classification.status_reading_bands") != "classification.status_reading_bands" else "Reading raster bands...")
            bands_data = []
            for b in range(self.total_bands):
                bands_data.append(self.reader.read_band(b))
                pct = 5 + int(25 * (b + 1) / self.total_bands)
                self.progress.emit(pct, f"Reading band {b + 1}/{self.total_bands}...")

            cube = np.stack(bands_data, axis=0)  # (bands, lines, samples)
            del bands_data  # Free intermediate list

            # 2. Run clustering algorithm (30% - 90%)
            if self.method_idx == 0:
                method_name = "K-Means"
                class_map, centers = kmeans_clustering(
                    cube,
                    num_classes=self.num_classes,
                    max_iter=self.max_iters,
                    progress_callback=lambda p, msg: self.progress.emit(p, msg),
                )
            else:
                method_name = "ISODATA"
                self.progress.emit(40, "Running ISODATA clustering...")
                class_map, centers = isodata_clustering(
                    cube,
                    initial_classes=self.num_classes,
                    max_iter=self.max_iters,
                )

            del cube  # Free cube working memory

            # 3. Generate thematic color rendering (90% - 100%)
            self.progress.emit(92, "Generating thematic RGB map...")
            thematic_rgb = create_thematic_rgb(class_map)
            layer_name = f"Classify ({method_name}, K={self.num_classes})"

            self.progress.emit(100, "Classification complete!")
            self.finished.emit(layer_name, class_map, self.reader.metadata, thematic_rgb)

        except Exception as e:
            self.failed.emit(str(e))


class ClassificationDialog(QDialog):
    """Dialog for unsupervised remote sensing classification."""

    result_generated = Signal(str, np.ndarray, object, object)  # layer_name, 2D class_map, meta, 3D thematic RGB array

    def __init__(self, layer: RasterLayer, reader, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self._worker: Optional[ClassificationWorker] = None
        self.setWindowTitle(tr("dialog.classification.title"))
        self.resize(440, 300)

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

        # Progress bar & status label (hidden until execution starts)
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

        # Buttons
        btn_box = QHBoxLayout()
        self.btn_run = QPushButton(tr("dialog.classification.btn_run"))
        self.btn_run.setStyleSheet("background-color: #238636; color: white; font-weight: bold; min-height: 28px;")
        self.btn_run.clicked.connect(self._run_classification)
        btn_box.addWidget(self.btn_run)

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_box.addWidget(self.btn_cancel)

        layout.addLayout(btn_box)

    def _run_classification(self) -> None:
        """Start background classification worker with real-time UI progress feedback."""
        method_idx = self.cb_method.currentIndex()
        num_classes = self.spin_classes.value()
        max_iters = self.spin_iters.value()

        # Update UI state
        self.btn_run.setEnabled(False)
        self.cb_method.setEnabled(False)
        self.spin_classes.setEnabled(False)
        self.spin_iters.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress.setVisible(True)
        self.lbl_progress.setText("Starting classification...")

        self._worker = ClassificationWorker(
            reader=self.reader,
            total_bands=self.layer.metadata.bands,
            method_idx=method_idx,
            num_classes=num_classes,
            max_iters=max_iters,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        """Handle background worker progress updates."""
        self.progress_bar.setValue(percent)
        self.lbl_progress.setText(message)

    @Slot(str, np.ndarray, object, object)
    def _on_finished(
        self,
        layer_name: str,
        class_map: np.ndarray,
        meta: object,
        thematic_rgb: np.ndarray,
    ) -> None:
        """Handle classification completion."""
        event_bus.status_message.emit(f"Classification completed: {layer_name}", 4000)
        self.result_generated.emit(layer_name, class_map, meta, thematic_rgb)
        self.accept()

    @Slot(str)
    def _on_failed(self, error_msg: str) -> None:
        """Handle classification failure."""
        self.btn_run.setEnabled(True)
        self.cb_method.setEnabled(True)
        self.spin_classes.setEnabled(True)
        self.spin_iters.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)
        QMessageBox.critical(
            self,
            tr("classification.err_title"),
            f"{tr('classification.err_msg')}\n{error_msg}",
        )

    def _on_cancel(self) -> None:
        """Cancel and terminate worker thread if active."""
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
        self.reject()

    def closeEvent(self, event) -> None:
        """Ensure worker thread terminates gracefully on dialog close."""
        self._on_cancel()
        super().closeEvent(event)
