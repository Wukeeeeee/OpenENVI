"""OpenENVI Synthetic Benchmark Dataset Generator Dialog.

Provides an interactive GUI to generate realistic multi-endmember hyperspectral
cubes (ENVI standard or GeoTIFF) with georeferencing and spectral features.
"""

import os
import tempfile
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from core.i18n import tr
from core.synthetic import (
    generate_synthetic_cube,
    write_envi_dataset,
    write_geotiff_dataset,
)


class SyntheticDataDialog(QDialog):
    """Dialog for creating synthetic hyperspectral benchmark datasets."""

    dataset_generated = Signal(str)  # Emits generated file path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialog.synthetic.title"))
        self.resize(460, 320)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()

        # Format
        self.cb_format = QComboBox()
        self.cb_format.addItems(["ENVI Standard (.hdr/.dat)", "GeoTIFF (.tif)"])
        form.addRow("File Format:", self.cb_format)

        # Dimensions
        self.spin_size = QSpinBox()
        self.spin_size.setRange(32, 512)
        self.spin_size.setValue(128)
        self.spin_size.setSingleStep(32)
        form.addRow(tr("dialog.synthetic.samples"), self.spin_size)

        # Bands
        self.spin_bands = QSpinBox()
        self.spin_bands.setRange(8, 224)
        self.spin_bands.setValue(64)
        self.spin_bands.setSingleStep(8)
        form.addRow(tr("dialog.synthetic.bands"), self.spin_bands)

        # Output Folder
        self.edit_dir = QLineEdit(tempfile.gettempdir())
        h_dir = QHBoxLayout()
        h_dir.addWidget(self.edit_dir)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse)
        h_dir.addWidget(btn_browse)
        form.addRow("Output Directory:", h_dir)

        layout.addLayout(form)
        layout.addStretch()

        # Action Buttons
        btn_box = QHBoxLayout()
        self.btn_gen = QPushButton(tr("dialog.synthetic.btn_generate"))
        self.btn_gen.setStyleSheet("background-color: #238636; color: white; font-weight: bold;")
        self.btn_gen.clicked.connect(self._generate)
        btn_box.addWidget(self.btn_gen)

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        layout.addLayout(btn_box)

    def _browse(self) -> None:
        """Select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.edit_dir.text())
        if dir_path:
            self.edit_dir.setText(dir_path)

    def _generate(self) -> None:
        """Generate synthetic dataset."""
        fmt_idx = self.cb_format.currentIndex()
        sz = self.spin_size.value()
        bands = self.spin_bands.value()
        out_dir = self.edit_dir.text().strip()

        if not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Invalid Directory", "Specified output directory does not exist.")
            return

        try:
            cube, wl, gt = generate_synthetic_cube(lines=sz, samples=sz, bands=bands)

            if fmt_idx == 0:
                # ENVI format
                base_path = os.path.join(out_dir, f"openenvi_benchmark_{sz}x{sz}_{bands}b")
                hdr_path, _ = write_envi_dataset(base_path, cube, wl)
                res_path = hdr_path
            else:
                # GeoTIFF
                tif_path = os.path.join(out_dir, f"openenvi_benchmark_{sz}x{sz}_{bands}b.tif")
                write_geotiff_dataset(tif_path, cube, wl)
                res_path = tif_path

            self.dataset_generated.emit(res_path)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Generation Failed", f"Failed to generate synthetic data: {e}")
