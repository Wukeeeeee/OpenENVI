"""OpenENVI Resize Data & Spatial/Spectral Subsetting Dialog.

Provides interactive spatial crop coordinates, resolution resizing,
and spectral band selection.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.algorithms.subset import resize_subset_raster
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import RasterLayer, RasterMetadata


class ResizeWorker(QThread):
    """Background worker for executing spatial/spectral subset and resize."""

    progress = Signal(int, int)
    finished = Signal(str, np.ndarray, object)
    failed = Signal(str)

    def __init__(
        self,
        reader: BaseRasterReader,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        selected_bands: List[int],
        scale_factor: float,
        resample_method: str,
        name: str,
    ):
        super().__init__()
        self.reader = reader
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.selected_bands = selected_bands
        self.scale_factor = scale_factor
        self.resample_method = resample_method
        self.name = name

    def run(self):
        try:
            cube, meta = resize_subset_raster(
                reader=self.reader,
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
                selected_bands=self.selected_bands,
                scale_factor=self.scale_factor,
                resample_method=self.resample_method,
                progress_callback=lambda c, t: self.progress.emit(c, t),
            )
            self.finished.emit(self.name, cube, meta)
        except Exception as e:
            self.failed.emit(str(e))


class ResizeDataDialog(QDialog):
    """Dialog for resizing data and spatial/spectral subsetting."""

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

        self.setWindowTitle(f"{tr('toolbox.tool_resize')} - {layer.name}")
        self.resize(550, 620)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        meta = self.layer.metadata
        w, h = meta.width, meta.height

        # 1. Spatial Subset Group
        grp_spatial = QGroupBox(tr("resize.grp_spatial"))
        layout_spatial = QVBoxLayout(grp_spatial)

        grid_coords = QHBoxLayout()

        # Samples (X)
        v_x = QVBoxLayout()
        v_x.addWidget(QLabel(tr("resize.samples")))
        h_x = QHBoxLayout()
        h_x.addWidget(QLabel(tr("resize.start")))
        self.sp_xmin = QSpinBox()
        self.sp_xmin.setRange(0, w - 1)
        self.sp_xmin.setValue(0)
        h_x.addWidget(self.sp_xmin)

        h_x.addWidget(QLabel(tr("resize.end")))
        self.sp_xmax = QSpinBox()
        self.sp_xmax.setRange(1, w)
        self.sp_xmax.setValue(w)
        h_x.addWidget(self.sp_xmax)
        v_x.addLayout(h_x)
        grid_coords.addLayout(v_x)

        # Lines (Y)
        v_y = QVBoxLayout()
        v_y.addWidget(QLabel(tr("resize.lines")))
        h_y = QHBoxLayout()
        h_y.addWidget(QLabel(tr("resize.start")))
        self.sp_ymin = QSpinBox()
        self.sp_ymin.setRange(0, h - 1)
        self.sp_ymin.setValue(0)
        h_y.addWidget(self.sp_ymin)

        h_y.addWidget(QLabel(tr("resize.end")))
        self.sp_ymax = QSpinBox()
        self.sp_ymax.setRange(1, h)
        self.sp_ymax.setValue(h)
        h_y.addWidget(self.sp_ymax)
        v_y.addLayout(h_y)
        grid_coords.addLayout(v_y)

        layout_spatial.addLayout(grid_coords)

        # Spatial subset buttons (Reset and ROI)
        h_btn_spatial = QHBoxLayout()
        btn_reset = QPushButton(tr("resize.btn_reset"))
        btn_reset.clicked.connect(self._reset_spatial)
        h_btn_spatial.addWidget(btn_reset)

        self.btn_from_roi = QPushButton(tr("resize.btn_from_roi") if tr("resize.btn_from_roi") != "resize.btn_from_roi" else "Subset by ROI...")
        self.btn_from_roi.clicked.connect(self._subset_from_roi)
        if not self.layer.rois:
            self.btn_from_roi.setToolTip("No ROIs available on this layer")
        h_btn_spatial.addWidget(self.btn_from_roi)
        layout_spatial.addLayout(h_btn_spatial)

        # Resample scale factor & interpolation method
        scale_box = QHBoxLayout()
        scale_box.addWidget(QLabel(tr("resize.scale_factor")))
        self.cb_scale = QComboBox()
        self.cb_scale.addItem(tr("resize.scale_1"), 1.0)
        self.cb_scale.addItem(tr("resize.scale_05"), 0.5)
        self.cb_scale.addItem(tr("resize.scale_025"), 0.25)
        self.cb_scale.addItem(tr("resize.scale_2"), 2.0)
        self.cb_scale.currentIndexChanged.connect(self._update_preview)
        scale_box.addWidget(self.cb_scale)

        scale_box.addWidget(QLabel(tr("resize.resample_method") if tr("resize.resample_method") != "resize.resample_method" else "Method:"))
        self.cb_method = QComboBox()
        self.cb_method.addItem(tr("resize.method_nearest") if tr("resize.method_nearest") != "resize.method_nearest" else "Nearest Neighbor", "nearest")
        self.cb_method.addItem(tr("resize.method_bilinear") if tr("resize.method_bilinear") != "resize.method_bilinear" else "Bilinear", "bilinear")
        self.cb_method.addItem(tr("resize.method_bicubic") if tr("resize.method_bicubic") != "resize.method_bicubic" else "Bicubic", "bicubic")
        self.cb_method.setCurrentIndex(1)  # Default Bilinear
        scale_box.addWidget(self.cb_method)

        layout_spatial.addLayout(scale_box)

        # Live dimension info preview
        self.lbl_dims_info = QLabel()
        self.lbl_dims_info.setStyleSheet("color: #3498db; font-weight: bold; padding: 2px;")
        layout_spatial.addWidget(self.lbl_dims_info)

        layout.addWidget(grp_spatial)

        # 2. Spectral Subset Group
        grp_spectral = QGroupBox(tr("resize.grp_spectral"))
        layout_spectral = QVBoxLayout(grp_spectral)

        self.list_bands = QListWidget()
        self.list_bands.setSelectionMode(QListWidget.MultiSelection)

        for b in range(meta.bands):
            b_name = f"Band {b + 1}"
            if meta.band_details and b < len(meta.band_details):
                binfo = meta.band_details[b]
                if binfo.wavelength is not None:
                    if binfo.name:
                        b_name = f"[{b + 1}] {binfo.name} ({binfo.wavelength:.1f} {binfo.wavelength_unit})"
                    else:
                        b_name = f"[{b + 1}] Band {b + 1} ({binfo.wavelength:.1f} {binfo.wavelength_unit})"
                elif binfo.name:
                    b_name = f"[{b + 1}] {binfo.name}"
            item = QListWidgetItem(b_name)
            item.setData(Qt.UserRole, b)
            self.list_bands.addItem(item)
            item.setSelected(True)

        self.list_bands.itemSelectionChanged.connect(self._update_preview)
        layout_spectral.addWidget(self.list_bands)

        btn_spectral_bar = QHBoxLayout()
        btn_all = QPushButton(tr("resize.btn_select_all"))
        btn_all.clicked.connect(self.list_bands.selectAll)
        btn_spectral_bar.addWidget(btn_all)

        btn_clear = QPushButton(tr("resize.btn_clear_all"))
        btn_clear.clicked.connect(self.list_bands.clearSelection)
        btn_spectral_bar.addWidget(btn_clear)
        layout_spectral.addLayout(btn_spectral_bar)

        layout.addWidget(grp_spectral)

        # 3. Output Name
        out_bar = QHBoxLayout()
        out_bar.addWidget(QLabel(tr("resize.out_name")))
        self.txt_out_name = QLineEdit(f"{self.layer.name}_Subset")
        out_bar.addWidget(self.txt_out_name)
        layout.addLayout(out_bar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Action buttons
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton(tr("dialog.btn_ok"))
        self.btn_ok.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_ok.clicked.connect(self._start_resize)
        btn_bar.addWidget(self.btn_ok)

        layout.addLayout(btn_bar)

        # Connect spinbox coordinate updates to live preview
        self.sp_xmin.valueChanged.connect(self._update_preview)
        self.sp_xmax.valueChanged.connect(self._update_preview)
        self.sp_ymin.valueChanged.connect(self._update_preview)
        self.sp_ymax.valueChanged.connect(self._update_preview)
        self._update_preview()

    def _update_preview(self):
        """Update output dimensions preview label."""
        xmin = self.sp_xmin.value()
        xmax = self.sp_xmax.value()
        ymin = self.sp_ymin.value()
        ymax = self.sp_ymax.value()
        scale = float(self.cb_scale.currentData()) if self.cb_scale.currentData() else 1.0

        cw = max(0, xmax - xmin)
        ch = max(0, ymax - ymin)
        tw = max(1, int(round(cw * scale))) if cw > 0 else 0
        th = max(1, int(round(ch * scale))) if ch > 0 else 0
        selected_count = len(self.list_bands.selectedItems())

        self.lbl_dims_info.setText(
            f"Original: {self.layer.metadata.width}x{self.layer.metadata.height} | "
            f"Crop: {cw}x{ch} | Output: {tw}x{th} ({scale}x), {selected_count} Bands"
        )

    def _reset_spatial(self):
        w = self.layer.metadata.width
        h = self.layer.metadata.height
        self.sp_xmin.setValue(0)
        self.sp_xmax.setValue(w)
        self.sp_ymin.setValue(0)
        self.sp_ymax.setValue(h)
        self._update_preview()

    def _subset_from_roi(self):
        """Set spatial subset bounds to the bounding box of a chosen ROI."""
        if not self.layer.rois:
            QMessageBox.information(
                self,
                tr("dialog.info") if tr("dialog.info") != "dialog.info" else "Info",
                tr("resize.no_rois_msg") if tr("resize.no_rois_msg") != "resize.no_rois_msg" else "No ROIs defined on current layer.",
            )
            return

        from PySide6.QtWidgets import QInputDialog
        items = [f"{r.name} ({r.color})" for r in self.layer.rois]
        choice, ok = QInputDialog.getItem(
            self,
            tr("resize.select_roi_title") if tr("resize.select_roi_title") != "resize.select_roi_title" else "Select ROI",
            tr("resize.select_roi_prompt") if tr("resize.select_roi_prompt") != "resize.select_roi_prompt" else "Choose ROI to define spatial bounding box:",
            items,
            0,
            False,
        )
        if ok and choice:
            idx = items.index(choice)
            roi = self.layer.rois[idx]
            w = self.layer.metadata.width
            h = self.layer.metadata.height

            if roi.bbox is not None:
                x0, y0, x1, y1 = roi.bbox
            elif roi.polygons:
                all_pts = [pt for poly in roi.polygons for pt in poly]
                if all_pts:
                    x0 = int(np.floor(min(p[0] for p in all_pts)))
                    x1 = int(np.ceil(max(p[0] for p in all_pts)))
                    y0 = int(np.floor(min(p[1] for p in all_pts)))
                    y1 = int(np.ceil(max(p[1] for p in all_pts)))
                else:
                    return
            else:
                return

            x0 = max(0, min(w - 1, x0))
            x1 = max(x0 + 1, min(w, x1))
            y0 = max(0, min(h - 1, y0))
            y1 = max(y0 + 1, min(h, y1))

            self.sp_xmin.setValue(x0)
            self.sp_xmax.setValue(x1)
            self.sp_ymin.setValue(y0)
            self.sp_ymax.setValue(y1)
            self._update_preview()

    def _start_resize(self):
        selected_items = self.list_bands.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self,
                tr("resize.err_no_bands_title"),
                tr("resize.err_no_bands_msg"),
            )
            return

        selected_bands = [item.data(Qt.UserRole) for item in selected_items]
        selected_bands.sort()

        xmin = self.sp_xmin.value()
        xmax = self.sp_xmax.value()
        ymin = self.sp_ymin.value()
        ymax = self.sp_ymax.value()

        if xmin >= xmax or ymin >= ymax:
            QMessageBox.warning(
                self,
                tr("resize.err_bounds_title"),
                tr("resize.err_bounds_msg"),
            )
            return

        scale_factor = float(self.cb_scale.currentData())
        resample_method = self.cb_method.currentData() or "bilinear"
        name = self.txt_out_name.text().strip() or f"{self.layer.name}_Subset"

        self.btn_ok.setEnabled(False)
        self.progress_bar.setRange(0, len(selected_bands))
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.worker = ResizeWorker(
            reader=self.reader,
            x_min=xmin,
            x_max=xmax,
            y_min=ymin,
            y_max=ymax,
            selected_bands=selected_bands,
            scale_factor=scale_factor,
            resample_method=resample_method,
            name=name,
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, name: str, cube: np.ndarray, meta: object):
        self.result_generated.emit(name, cube, meta)
        self.accept()

    def _on_failed(self, err: str):
        self.btn_ok.setEnabled(True)
        self.progress_bar.hide()
        QMessageBox.critical(self, tr("dialog.error"), f"{tr('resize.err_failed')}: {err}")
