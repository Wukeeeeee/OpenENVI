"""OpenENVI Principal Component Analysis (PCA) Dialog.

Computes PCA transforms for dimensionality reduction and generates
eigenvector/variance reports and multi-component composite layers.
"""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from core.algorithms.spectral import compute_pca
from core.i18n import tr
from core.models import RasterLayer


class PCADialog(QDialog):
    """Dialog for computing Principal Component Analysis."""

    result_generated = Signal(str, np.ndarray)  # layer_name, 3D array (components, lines, samples)

    def __init__(self, layer: RasterLayer, reader, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self.setWindowTitle(tr("dialog.pca.title"))
        self.resize(460, 360)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Form Controls
        form = QFormLayout()
        self.spin_components = QSpinBox()
        self.spin_components.setRange(1, min(10, layer.metadata.bands))
        self.spin_components.setValue(min(3, layer.metadata.bands))
        form.addRow(tr("dialog.pca.num_comp"), self.spin_components)

        self.chk_rgb = QCheckBox(tr("dialog.pca.chk_rgb"))
        self.chk_rgb.setChecked(True)
        form.addRow(self.chk_rgb)
        layout.addLayout(form)

        # Report / Output summary
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setPlaceholderText(tr("dialog.pca.report_tip"))
        layout.addWidget(self.txt_report, stretch=1)

        # Buttons
        btn_box = QHBoxLayout()
        self.btn_run = QPushButton(tr("dialog.pca.btn_run"))
        self.btn_run.setStyleSheet("background-color: #238636; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self._run_pca)
        btn_box.addWidget(self.btn_run)

        self.btn_close = QPushButton(tr("dialog.btn_ok"))
        self.btn_close.clicked.connect(self.accept)
        btn_box.addWidget(self.btn_close)

        layout.addLayout(btn_box)

    def _run_pca(self) -> None:
        """Execute PCA algorithm."""
        num_comp = self.spin_components.value()
        try:
            self.btn_run.setEnabled(False)
            meta = self.layer.metadata
            total_pixels = meta.lines * meta.samples
            bands_count = meta.bands

            if total_pixels > 500_000:
                # Fast sampled covariance on large satellite imagery (> 500k pixels)
                step = max(1, int(np.sqrt(total_pixels / 250_000)))
                sample_bands = [self.reader.read_band(b)[::step, ::step] for b in range(bands_count)]
                sample_cube = np.stack(sample_bands, axis=0)
                sample_flat = sample_cube.reshape(bands_count, -1).T.astype(np.float32)
                mean_vec = np.mean(sample_flat, axis=0)
                cov = np.cov(sample_flat - mean_vec, rowvar=False)

                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                total_var = np.sum(eigenvalues)
                explained_var = (eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues))[:num_comp]

                # Stream project components one by one without multi-gigabyte memory spikes
                top_eigenvectors = eigenvectors[:, :num_comp]
                score_cube = np.zeros((num_comp, meta.lines, meta.samples), dtype=np.float32)

                for k in range(num_comp):
                    for b in range(bands_count):
                        coeff = float(top_eigenvectors[b, k])
                        if abs(coeff) > 1e-7:
                            b_data = self.reader.read_band(b)
                            score_cube[k] += coeff * (b_data - float(mean_vec[b]))
                            del b_data
            else:
                # In-memory execution for smaller benchmark cubes
                bands_data = [self.reader.read_band(b) for b in range(bands_count)]
                cube = np.stack(bands_data, axis=0)
                score_cube, eigenvalues, explained_var = compute_pca(cube, num_components=num_comp)

            # Build report
            lines = [f"=== PCA Transformation Report for {self.layer.name} ===", ""]
            lines.append(f"Input Bands: {self.layer.metadata.bands}")
            lines.append(f"Output Components: {num_comp}")
            lines.append("-" * 45)
            for i in range(num_comp):
                pct = explained_var[i] * 100.0
                lines.append(f"PC {i + 1}: Eigenvalue = {eigenvalues[i]:.4e} | Variance = {pct:.2f}%")
            cum_var = np.sum(explained_var[:num_comp]) * 100.0
            lines.append("-" * 45)
            lines.append(f"Cumulative Variance Explained: {cum_var:.2f}%")
            self.txt_report.setPlainText("\n".join(lines))

            if self.chk_rgb.isChecked() and num_comp >= 3:
                # Transpose to (lines, samples, 3)
                rgb_composite = np.transpose(score_cube[:3], (1, 2, 0))
                self.result_generated.emit(f"PCA (PC1-PC2-PC3)", rgb_composite)
            else:
                for i in range(num_comp):
                    self.result_generated.emit(f"PCA Band (PC{i + 1})", score_cube[i])

            self.btn_run.setEnabled(False)
        except Exception as e:
            self.btn_run.setEnabled(True)
            QMessageBox.critical(self, "PCA Computation Failed", f"Error during PCA computation:\n{e}")
