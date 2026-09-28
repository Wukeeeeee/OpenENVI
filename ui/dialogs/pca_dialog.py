"""OpenENVI Principal Component Analysis (PCA) & Minimum Noise Fraction (MNF) Dialog.

Computes PCA or MNF transforms for dimensionality reduction, noise segregation,
eigenvalue/variance/SNR reports, and multi-component composite layers.
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

from core.algorithms.spectral import compute_mnf, compute_pca
from core.i18n import tr
from core.models import RasterLayer


class PCADialog(QDialog):
    """Dialog for computing Principal Component Analysis (PCA) or Minimum Noise Fraction (MNF)."""

    result_generated = Signal(str, np.ndarray, object)  # layer_name, array, parent_meta

    def __init__(self, layer: RasterLayer, reader, mode: str = "pca", parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self.mode = mode.lower()  # "pca" or "mnf"

        is_mnf = self.mode == "mnf"
        title = tr("dialog.mnf.title") if is_mnf else tr("dialog.pca.title")
        self.setWindowTitle(f"{title} - {layer.name}")
        self.resize(500, 420)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Form Controls
        form = QFormLayout()
        self.spin_components = QSpinBox()
        self.spin_components.setRange(1, min(20, layer.metadata.bands))
        self.spin_components.setValue(min(3, layer.metadata.bands))
        num_lbl = tr("dialog.mnf.num_comp") if is_mnf else tr("dialog.pca.num_comp")
        form.addRow(num_lbl, self.spin_components)

        rgb_lbl = tr("dialog.mnf.chk_rgb") if is_mnf else tr("dialog.pca.chk_rgb")
        self.chk_rgb = QCheckBox(rgb_lbl)
        self.chk_rgb.setChecked(True)
        form.addRow(self.chk_rgb)
        layout.addLayout(form)

        # Report / Output summary
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        tip = tr("dialog.mnf.report_tip") if is_mnf else tr("dialog.pca.report_tip")
        self.txt_report.setPlaceholderText(tip)
        layout.addWidget(self.txt_report, stretch=1)

        # Buttons
        btn_box = QHBoxLayout()
        run_lbl = tr("dialog.mnf.btn_run") if is_mnf else tr("dialog.pca.btn_run")
        self.btn_run = QPushButton(run_lbl)
        self.btn_run.setStyleSheet("background-color: #238636; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self._run_transform)
        btn_box.addWidget(self.btn_run)

        self.btn_close = QPushButton(tr("dialog.btn_ok"))
        self.btn_close.clicked.connect(self.accept)
        btn_box.addWidget(self.btn_close)

        layout.addLayout(btn_box)

    def _run_transform(self) -> None:
        """Execute PCA or MNF algorithm."""
        num_comp = self.spin_components.value()
        try:
            self.btn_run.setEnabled(False)
            meta = self.layer.metadata
            total_pixels = meta.lines * meta.samples
            bands_count = meta.bands

            if self.mode == "mnf":
                # MNF Transform
                bands_data = [self.reader.read_band(b) for b in range(bands_count)]
                cube = np.stack(bands_data, axis=0)  # (bands, lines, samples)
                score_cube, eigenvalues = compute_mnf(cube, num_components=num_comp)
                del cube

                # Build MNF Report
                lines = [f"=== Minimum Noise Fraction (MNF) Report for {self.layer.name} ===", ""]
                lines.append(f"Input Bands: {bands_count}")
                lines.append(f"Output Components: {num_comp}")
                lines.append("-" * 50)
                for i in range(num_comp):
                    eig = float(eigenvalues[i])
                    snr = max(0.0, eig - 1.0)
                    lines.append(f"MNF Band {i + 1}: Eigenvalue = {eig:.4f} | Estimated SNR = {snr:.2f}")
                lines.append("-" * 50)
                lines.append("Note: MNF Eigenvalues equal (Signal-to-Noise Ratio + 1).")
                lines.append("Bands with eigenvalues near 1.0 contain primarily noise.")
                self.txt_report.setPlainText("\n".join(lines))

                prefix = "MNF"
            else:
                # PCA Transform
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
                    bands_data = [self.reader.read_band(b) for b in range(bands_count)]
                    cube = np.stack(bands_data, axis=0)
                    score_cube, eigenvalues, explained_var = compute_pca(cube, num_components=num_comp)
                    del cube

                # Build PCA Report
                lines = [f"=== PCA Transformation Report for {self.layer.name} ===", ""]
                lines.append(f"Input Bands: {bands_count}")
                lines.append(f"Output Components: {num_comp}")
                lines.append("-" * 45)
                for i in range(num_comp):
                    pct = explained_var[i] * 100.0
                    lines.append(f"PC {i + 1}: Eigenvalue = {eigenvalues[i]:.4e} | Variance = {pct:.2f}%")
                cum_var = np.sum(explained_var[:num_comp]) * 100.0
                lines.append("-" * 45)
                lines.append(f"Cumulative Variance Explained: {cum_var:.2f}%")
                self.txt_report.setPlainText("\n".join(lines))

                prefix = "PCA"

            # Emit layers
            if self.chk_rgb.isChecked() and num_comp >= 3:
                rgb_composite = np.transpose(score_cube[:3], (1, 2, 0))
                self.result_generated.emit(f"{prefix} ({prefix}1-{prefix}2-{prefix}3)", rgb_composite, meta)
            else:
                for i in range(num_comp):
                    self.result_generated.emit(f"{prefix} Band ({prefix}{i + 1})", score_cube[i], meta)

            self.btn_run.setEnabled(False)
        except Exception as e:
            self.btn_run.setEnabled(True)
            err_title = tr("dialog.mnf.err_title") if self.mode == "mnf" else tr("pca.err_title")
            err_msg = tr("dialog.mnf.err_msg") if self.mode == "mnf" else tr("pca.err_msg")
            QMessageBox.critical(self, err_title, f"{err_msg}\n{e}")
