"""OpenENVI Color Space Transform Dialog.

Provides interactive conversion between RGB, HSV, and Grayscale.
"""

from typing import Optional
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.algorithms.color import hsv_to_rgb, rgb_to_grayscale, rgb_to_hsv
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterLayer, RasterMetadata


class ColorWorker(QThread):
    """Background worker for color transformation."""

    finished = Signal(str, np.ndarray, object)
    failed = Signal(str)

    def __init__(
        self,
        reader: BaseRasterReader,
        r_idx: int,
        g_idx: int,
        b_idx: int,
        mode: str,
        normalize: bool,
        name: str,
    ):
        super().__init__()
        self.reader = reader
        self.r_idx = r_idx
        self.g_idx = g_idx
        self.b_idx = b_idx
        self.mode = mode
        self.normalize = normalize
        self.name = name

    def run(self):
        try:
            r = self.reader.read_band(self.r_idx)
            g = self.reader.read_band(self.g_idx)
            b = self.reader.read_band(self.b_idx)

            if self.mode == "RGB_TO_HSV":
                cube, channel_names = rgb_to_hsv(r, g, b, normalize_inputs=self.normalize)
            elif self.mode == "HSV_TO_RGB":
                cube, channel_names = hsv_to_rgb(r, g, b)
            else:
                cube, channel_names = rgb_to_grayscale(r, g, b)

            band_details = [
                BandInfo(index=i, name=c_name)
                for i, c_name in enumerate(channel_names)
            ]

            meta = RasterMetadata(
                width=self.reader.metadata.width,
                height=self.reader.metadata.height,
                bands=len(channel_names),
                dtype="float32",
                crs=self.reader.metadata.crs,
                transform=self.reader.metadata.transform,
                band_details=band_details,
                raw_header=self.reader.metadata.raw_header,
            )
            self.finished.emit(self.name, cube, meta)
        except Exception as e:
            self.failed.emit(str(e))


class ColorTransformDialog(QDialog):
    """Dialog for performing color space transformations."""

    result_generated = Signal(str, np.ndarray, object)

    def __init__(
        self,
        layer: RasterLayer,
        reader: BaseRasterReader,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader

        self.setWindowTitle(f"{tr('toolbox.tool_color')} - {layer.name}")
        self.resize(500, 420)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Transform mode
        grp_mode = QGroupBox("Transformation Mode")
        l_mode = QVBoxLayout(grp_mode)
        self.cb_mode = QComboBox()
        self.cb_mode.addItem("RGB -> HSV (Hue, Saturation, Value)", "RGB_TO_HSV")
        self.cb_mode.addItem("HSV -> RGB (Red, Green, Blue)", "HSV_TO_RGB")
        self.cb_mode.addItem("RGB -> Grayscale Luminance", "RGB_TO_GRAY")
        self.cb_mode.currentIndexChanged.connect(self._on_mode_changed)
        l_mode.addWidget(self.cb_mode)
        layout.addWidget(grp_mode)

        # 2. Channel Assignment
        self.grp_channels = QGroupBox("Input Channel Assignment")
        l_chan = QVBoxLayout(self.grp_channels)

        # Row 1
        h1 = QHBoxLayout()
        self.lbl_ch1 = QLabel("Red Band (R):")
        self.cb_ch1 = QComboBox()
        h1.addWidget(self.lbl_ch1)
        h1.addWidget(self.cb_ch1, stretch=1)
        l_chan.addLayout(h1)

        # Row 2
        h2 = QHBoxLayout()
        self.lbl_ch2 = QLabel("Green Band (G):")
        self.cb_ch2 = QComboBox()
        h2.addWidget(self.lbl_ch2)
        h2.addWidget(self.cb_ch2, stretch=1)
        l_chan.addLayout(h2)

        # Row 3
        h3 = QHBoxLayout()
        self.lbl_ch3 = QLabel("Blue Band (B):")
        self.cb_ch3 = QComboBox()
        h3.addWidget(self.lbl_ch3)
        h3.addWidget(self.cb_ch3, stretch=1)
        l_chan.addLayout(h3)

        self._populate_bands()
        layout.addWidget(self.grp_channels)

        # Options
        self.chk_norm = QCheckBox("Normalize raw input channels to [0.0, 1.0] dynamic range")
        self.chk_norm.setChecked(True)
        layout.addWidget(self.chk_norm)

        # Output Name
        h_out = QHBoxLayout()
        h_out.addWidget(QLabel("Output Layer Name:"))
        self.txt_out_name = QLineEdit(f"{self.layer.name}_HSV")
        h_out.addWidget(self.txt_out_name)
        layout.addLayout(h_out)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton(tr("dialog.btn_ok"))
        self.btn_ok.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_ok.clicked.connect(self._start_transform)
        btn_bar.addWidget(self.btn_ok)

        layout.addLayout(btn_bar)

    def _populate_bands(self):
        meta = self.layer.metadata
        for b in range(meta.bands):
            b_name = (
                meta.band_details[b].name
                if meta.band_details and b < len(meta.band_details) and meta.band_details[b].name
                else f"Band {b + 1}"
            )
            self.cb_ch1.addItem(b_name, b)
            self.cb_ch2.addItem(b_name, b)
            self.cb_ch3.addItem(b_name, b)

        # Set sensible defaults
        total = meta.bands
        self.cb_ch1.setCurrentIndex(0)
        self.cb_ch2.setCurrentIndex(min(1, total - 1))
        self.cb_ch3.setCurrentIndex(min(2, total - 1))

    def _on_mode_changed(self):
        mode = self.cb_mode.currentData()
        if mode == "HSV_TO_RGB":
            self.lbl_ch1.setText("Hue Band (H):")
            self.lbl_ch2.setText("Saturation Band (S):")
            self.lbl_ch3.setText("Value Band (V):")
            self.txt_out_name.setText(f"{self.layer.name}_RGB")
            self.chk_norm.setChecked(False)
        elif mode == "RGB_TO_GRAY":
            self.lbl_ch1.setText("Red Band (R):")
            self.lbl_ch2.setText("Green Band (G):")
            self.lbl_ch3.setText("Blue Band (B):")
            self.txt_out_name.setText(f"{self.layer.name}_Gray")
            self.chk_norm.setChecked(False)
        else:
            self.lbl_ch1.setText("Red Band (R):")
            self.lbl_ch2.setText("Green Band (G):")
            self.lbl_ch3.setText("Blue Band (B):")
            self.txt_out_name.setText(f"{self.layer.name}_HSV")
            self.chk_norm.setChecked(True)

    def _start_transform(self):
        r_idx = self.cb_ch1.currentData()
        g_idx = self.cb_ch2.currentData()
        b_idx = self.cb_ch3.currentData()
        mode = self.cb_mode.currentData()
        normalize = self.chk_norm.isChecked()
        name = self.txt_out_name.text().strip() or f"{self.layer.name}_ColorTransform"

        self.btn_ok.setEnabled(False)
        self.progress_bar.show()

        self.worker = ColorWorker(
            reader=self.reader,
            r_idx=r_idx,
            g_idx=g_idx,
            b_idx=b_idx,
            mode=mode,
            normalize=normalize,
            name=name,
        )
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, name: str, cube: np.ndarray, meta: object):
        self.result_generated.emit(name, cube, meta)
        self.accept()

    def _on_failed(self, err: str):
        self.btn_ok.setEnabled(True)
        self.progress_bar.hide()
        QMessageBox.critical(self, "Transform Failed", f"Color transform failed: {err}")
