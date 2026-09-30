"""OpenENVI Main Window.

Assembles the dockable workspace modeled after ENVI Classic and ENVI 5.x,
integrating Layer Manager, Data Manager, Toolbox, Spectral Profile, Main View,
interactive algorithm dialogs, and real-time cursor/spectral probing.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from PySide6.QtCore import QSettings, Qt, Slot, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QToolBar,
)

from core.events import event_bus
from core.i18n import i18n, tr
from core.io.base import BaseRasterReader
from core.io.memory import MemoryRasterReader
from core.io.reader import open_raster
from core.models import RasterLayer
from ui.data_manager import DataManagerDock
from ui.dialogs.accuracy_dialog import AccuracyAssessmentDialog
from ui.dialogs.band_math_dialog import BandMathDialog
from ui.dialogs.classification_dialog import ClassificationDialog
from ui.dialogs.color_dialog import ColorTransformDialog
from ui.dialogs.continuum_dialog import ContinuumRemovalDialog
from ui.dialogs.export_dialog import ExportRasterDialog
from ui.dialogs.indices_dialog import IndicesDialog
from ui.dialogs.maxlik_dialog import MaximumLikelihoodDialog
from ui.dialogs.mosaic_dialog import MosaicDialog
from ui.dialogs.pansharpen_dialog import PanSharpenDialog
from ui.dialogs.pca_dialog import PCADialog
from ui.dialogs.radiometry_dialog import RadiometryDialog
from ui.dialogs.resize_dialog import ResizeDataDialog
from ui.dialogs.roi_dialog import ROIToolDialog
from ui.dialogs.sam_dialog import SAMDialog
from ui.dialogs.sid_dialog import SIDDialog
from ui.dialogs.sff_dialog import SFFDialog
from ui.dialogs.stacking_dialog import LayerStackingDialog
from ui.dialogs.stats_dialog import QuickStatsDialog
from ui.dialogs.svm_dialog import SVMDialog
from ui.dialogs.synthetic_dialog import SyntheticDataDialog
from ui.layer_manager import LayerManagerDock
from ui.main_view import MainViewWidget
from ui.spectral_profile import SpectralProfileDock
from ui.status_bar import OpenENVIStatusBar
from ui.toolbox import ToolboxDock


class OpenENVIMainWindow(QMainWindow):
    """Primary application window hosting the ENVI-style dockable workspace."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(1366, 820)
        self.setDockNestingEnabled(True)

        # Storage for open datasets and layers
        self._readers: Dict[str, BaseRasterReader] = {}
        self._layers: Dict[str, RasterLayer] = {}
        self._active_layer_id: Optional[str] = None

        # Pixel hover debounce timer (16ms) to prevent Landsat/multiband I/O storms
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(16)
        self._hover_timer.timeout.connect(self._do_pixel_hover)
        self._pending_hover_coords: Optional[Tuple[int, int]] = None

        # Restore saved language preference before building UI
        settings = QSettings("OpenENVI", "OpenENVI")
        saved_lang = settings.value("language", None)
        if saved_lang in ("en", "zh"):
            i18n.set_language(saved_lang)

        # Initialize Central Viewport
        self.main_view = MainViewWidget(self)
        self.setCentralWidget(self.main_view)

        # Initialize Status Bar
        self.status_bar = OpenENVIStatusBar(self)
        self.setStatusBar(self.status_bar)

        # Initialize Docks
        self._init_docks()

        # Build Actions, Menus, and Toolbar
        self._init_actions()
        self._init_menus()
        self._init_toolbar()

        # Connect UI and Data signals
        self._connect_signals()

        # Retranslate on creation and connect language changes
        self.retranslate_ui()
        i18n.language_changed.connect(lambda _: self.retranslate_ui())

        # Restore window geometry, dock layout, and stretch mode
        saved_geom = settings.value("geometry")
        if saved_geom:
            self.restoreGeometry(saved_geom)
        saved_state = settings.value("windowState")
        if saved_state:
            self.restoreState(saved_state)
        saved_stretch = settings.value("stretch_mode")
        if saved_stretch:
            idx = self.cb_stretch.findText(str(saved_stretch))
            if idx >= 0:
                self.cb_stretch.setCurrentIndex(idx)

        event_bus.status_message.emit(tr("app.initialized"), 3000)

    def _init_docks(self) -> None:
        """Create and place all 4 dockable panels."""
        # Left Side: Layer Manager (top) and Data Manager (bottom)
        self.dock_layer_manager = LayerManagerDock(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_layer_manager)

        self.dock_data_manager = DataManagerDock(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_data_manager)

        # Right Side: ENVI Toolbox
        self.dock_toolbox = ToolboxDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_toolbox)

        # Bottom Side: Spectral Profile (Z-Profile)
        self.dock_spectral_profile = SpectralProfileDock(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_spectral_profile)

    def _init_actions(self) -> None:
        """Create standard application actions."""
        # File Actions
        self.act_open = QAction("Open...", self)
        self.act_open.setShortcut(QKeySequence.Open)
        self.act_open.triggered.connect(self._on_open_file_dialog)

        self.act_gen_data = QAction("Generate Synthetic Benchmark Data...", self)
        self.act_gen_data.triggered.connect(self.show_synthetic_generator_dialog)

        self.act_exit = QAction("Exit", self)
        self.act_exit.setShortcut(QKeySequence.Quit)
        self.act_exit.triggered.connect(self.close)

        # Navigation Actions
        self.act_probe = QAction("Probe", self)
        self.act_probe.setCheckable(True)
        self.act_probe.setChecked(True)

        self.act_pan = QAction("Pan", self)
        self.act_pan.setCheckable(True)

        self.act_zoom_in = QAction("Zoom In", self)
        self.act_zoom_in.setShortcut(QKeySequence.ZoomIn)
        self.act_zoom_in.triggered.connect(self.main_view.zoom_in)

        self.act_zoom_out = QAction("Zoom Out", self)
        self.act_zoom_out.setShortcut(QKeySequence.ZoomOut)
        self.act_zoom_out.triggered.connect(self.main_view.zoom_out)

        self.act_zoom_fit = QAction("Fit", self)
        self.act_zoom_fit.triggered.connect(self.main_view.zoom_fit)

        self.act_toggle_overview = QAction(tr("main_view.overview_toggle"), self)
        self.act_toggle_overview.setCheckable(True)
        self.act_toggle_overview.setChecked(True)
        self.act_toggle_overview.toggled.connect(self.main_view.set_overview_visible)

        # Layer Actions
        self.act_layer_props = QAction("Layer Properties...", self)
        self.act_remove_layer = QAction("Remove Active Layer", self)
        self.act_remove_layer.triggered.connect(self._remove_active_layer)

        # Tools Actions
        self.act_band_math = QAction("Band Math...", self)
        self.act_band_math.triggered.connect(self.show_band_math_dialog)

        self.act_indices = QAction("Spectral Indices...", self)
        self.act_indices.triggered.connect(self.show_indices_dialog)

        self.act_pca = QAction("Principal Components Analysis (PCA)...", self)
        self.act_pca.triggered.connect(self.show_pca_dialog)

        self.act_mnf = QAction("Minimum Noise Fraction (MNF)...", self)
        self.act_mnf.triggered.connect(self.show_mnf_dialog)

        self.act_ica = QAction("Independent Component Analysis (ICA)...", self)
        self.act_ica.triggered.connect(self.show_ica_dialog)

        self.act_sam = QAction("Spectral Angle Mapper (SAM)...", self)
        self.act_sam.triggered.connect(self.show_sam_dialog)

        self.act_sid = QAction("Spectral Information Divergence (SID)...", self)
        self.act_sid.triggered.connect(self.show_sid_dialog)

        self.act_classification = QAction("Image Classification...", self)
        self.act_classification.triggered.connect(self.show_classification_dialog)

        self.act_maxlik = QAction("Maximum Likelihood Classification...", self)
        self.act_maxlik.triggered.connect(self.show_maxlik_dialog)

        self.act_svm = QAction("Support Vector Machine (SVM) Classification...", self)
        self.act_svm.triggered.connect(self.show_svm_dialog)

        self.act_roi = QAction("Region of Interest (ROI) Tool...", self)
        self.act_roi.triggered.connect(self.show_roi_dialog)

        self.act_export_raster = QAction("Export Raster / Save As...", self)
        self.act_export_raster.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.act_export_raster.triggered.connect(lambda: self.show_export_dialog())

        self.act_save_view_image = QAction("Save Viewport as Image...", self)
        self.act_save_view_image.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_view_image.triggered.connect(self._save_view_image)

        self.act_pansharpen = QAction("Pan-Sharpening Image Fusion...", self)
        self.act_pansharpen.triggered.connect(self.show_pansharpen_dialog)

        self.act_radiometry = QAction("Radiometric Calibration & Atmospheric Correction...", self)
        self.act_radiometry.triggered.connect(self.show_radiometry_dialog)

        self.act_stats = QAction("Quick Statistics...", self)
        self.act_stats.triggered.connect(self.show_stats_dialog)

        self.act_stacking = QAction("Layer Stacking...", self)
        self.act_stacking.triggered.connect(self.show_layer_stacking_dialog)

        self.act_mosaic = QAction("Mosaicking...", self)
        self.act_mosaic.triggered.connect(self.show_mosaic_dialog)

        self.act_resize = QAction("Resize Data (Spatial/Spectral)...", self)
        self.act_resize.triggered.connect(self.show_resize_dialog)

        self.act_color = QAction("Color Space Transforms (RGB-HSV)...", self)
        self.act_color.triggered.connect(self.show_color_transform_dialog)

        self.act_continuum = QAction("Continuum Removal...", self)
        self.act_continuum.triggered.connect(self.show_continuum_removal_dialog)

        self.act_accuracy = QAction("Confusion Matrix & Accuracy Assessment...", self)
        self.act_accuracy.triggered.connect(self.show_accuracy_assessment_dialog)

        # Window & Help Actions
        self.act_reset_layout = QAction("Reset Dock Layout", self)
        self.act_reset_layout.triggered.connect(self._reset_dock_layout)

        self.act_shortcuts = QAction("User Guide & Shortcuts...", self)
        self.act_shortcuts.triggered.connect(self._on_shortcuts)

        self.act_about = QAction("About OpenENVI", self)
        self.act_about.triggered.connect(self._on_about)

        # Language Actions (Mutual exclusion group)
        self.lang_group = QActionGroup(self)
        self.lang_group.setExclusive(True)

        self.act_lang_en = QAction("English", self)
        self.act_lang_en.setCheckable(True)
        self.act_lang_en.setChecked(i18n.current_language == "en")
        self.act_lang_en.triggered.connect(lambda: self._set_language("en"))
        self.lang_group.addAction(self.act_lang_en)

        self.act_lang_zh = QAction("简体中文 (Chinese)", self)
        self.act_lang_zh.setCheckable(True)
        self.act_lang_zh.setChecked(i18n.current_language == "zh")
        self.act_lang_zh.triggered.connect(lambda: self._set_language("zh"))
        self.lang_group.addAction(self.act_lang_zh)

    def _init_menus(self) -> None:
        """Build ENVI menu bar structure."""
        mb = self.menuBar()

        # File Menu
        self.menu_file = mb.addMenu("&File")
        self.menu_file.addAction(self.act_open)
        self.menu_file.addAction(self.act_export_raster)
        self.menu_file.addAction(self.act_save_view_image)
        self.menu_file.addAction(self.act_gen_data)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.act_exit)

        # View Menu
        self.menu_view = mb.addMenu("&View")
        self.menu_view.addAction(self.act_zoom_in)
        self.menu_view.addAction(self.act_zoom_out)
        self.menu_view.addAction(self.act_zoom_fit)
        self.menu_view.addSeparator()
        self.menu_view.addAction(self.dock_layer_manager.toggleViewAction())
        self.menu_view.addAction(self.dock_data_manager.toggleViewAction())
        self.menu_view.addAction(self.dock_toolbox.toggleViewAction())
        self.menu_view.addAction(self.dock_spectral_profile.toggleViewAction())
        self.menu_view.addSeparator()
        self.menu_view.addAction(self.act_toggle_overview)

        # Layer Menu
        self.menu_layer = mb.addMenu("&Layer")
        self.menu_layer.addAction(self.act_layer_props)
        self.menu_layer.addAction(self.act_remove_layer)

        # Display Menu
        self.menu_display = mb.addMenu("&Display")
        self.act_stretch_lin2 = self.menu_display.addAction("Linear 2% Stretch", lambda: self._on_stretch_changed("Linear 2%"))
        self.act_stretch_lin5 = self.menu_display.addAction("Linear 5% Stretch", lambda: self._on_stretch_changed("Linear 5%"))
        self.act_stretch_eq = self.menu_display.addAction("Equalization Stretch", lambda: self._on_stretch_changed("Equalization"))
        self.act_stretch_gauss = self.menu_display.addAction("Gaussian Stretch", lambda: self._on_stretch_changed("Gaussian"))

        # Tools Menu
        self.menu_tools = mb.addMenu("&Tools")
        self.menu_tools.addAction(self.act_roi)
        self.menu_tools.addAction(self.act_band_math)
        self.menu_tools.addAction(self.act_indices)
        self.menu_tools.addAction(self.act_pca)
        self.menu_tools.addAction(self.act_mnf)
        self.menu_tools.addAction(self.act_ica)
        self.menu_tools.addAction(self.act_sam)
        self.menu_tools.addAction(self.act_sid)
        self.menu_tools.addAction(self.act_classification)
        self.menu_tools.addAction(self.act_maxlik)
        self.menu_tools.addAction(self.act_svm)
        self.menu_tools.addAction(self.act_pansharpen)
        self.menu_tools.addAction(self.act_radiometry)
        self.menu_tools.addAction(self.act_stats)
        self.menu_tools.addAction(self.act_stacking)
        self.menu_tools.addAction(self.act_mosaic)
        self.menu_tools.addAction(self.act_resize)
        self.menu_tools.addAction(self.act_color)
        self.menu_tools.addAction(self.act_continuum)
        self.menu_tools.addAction(self.act_accuracy)

        # Window Menu
        self.menu_window = mb.addMenu("&Window")
        self.menu_window.addAction(self.act_reset_layout)

        # Language Menu
        self.menu_language = mb.addMenu("&Language")
        self.menu_language.addAction(self.act_lang_en)
        self.menu_language.addAction(self.act_lang_zh)

        # Help Menu
        self.menu_help = mb.addMenu("&Help")
        self.menu_help.addAction(self.act_shortcuts)
        self.menu_help.addSeparator()
        self.menu_help.addAction(self.act_about)

    def _init_toolbar(self) -> None:
        """Build primary ENVI-style toolbar."""
        tb = self.addToolBar("Main Toolbar")
        tb.setObjectName("MainToolBar")
        tb.setMovable(False)

        tb.addAction(self.act_open)
        tb.addSeparator()

        tb.addAction(self.act_probe)
        tb.addAction(self.act_pan)
        tb.addSeparator()

        tb.addAction(self.act_zoom_in)
        tb.addAction(self.act_zoom_out)
        tb.addAction(self.act_zoom_fit)
        tb.addSeparator()

        # Stretch Enhancement Selector
        self.lbl_stretch = QLabel(" Stretch: ")
        self.lbl_stretch.setStyleSheet("color: #8e9297; font-weight: bold;")
        tb.addWidget(self.lbl_stretch)

        self.cb_stretch = QComboBox()
        self.cb_stretch.addItems([
            "Linear 2%",
            "Linear 5%",
            "Equalization",
            "Gaussian",
            "No Stretch",
        ])
        self.cb_stretch.currentTextChanged.connect(self._on_stretch_changed)
        tb.addWidget(self.cb_stretch)

    def _connect_signals(self) -> None:
        """Wire UI interaction signals between docks, viewport, and status bar."""
        # Data Manager -> Viewport loading
        self.dock_data_manager.load_grayscale_requested.connect(self.load_grayscale_band)
        self.dock_data_manager.load_rgb_requested.connect(self.load_rgb_composition)
        self.dock_data_manager.close_file_requested.connect(self._on_layer_removed)

        # Layer Manager interactions
        self.dock_layer_manager.layer_visibility_changed.connect(self._on_layer_visibility_changed)
        self.dock_layer_manager.layer_removed.connect(self._on_layer_removed)
        self.dock_layer_manager.export_layer_requested.connect(self.show_export_dialog)
        self.dock_layer_manager.roi_tool_requested.connect(self.show_roi_dialog)
        self.dock_layer_manager.roi_visibility_changed.connect(self._on_roi_visibility_changed)
        self.dock_layer_manager.roi_removed.connect(self._on_roi_removed)
        event_bus.layer_changed.connect(self._on_active_layer_changed)

        # Toolbox double click -> Launch tool
        self.dock_toolbox.tool_selected.connect(self._on_toolbox_tool_selected)

        # Canvas -> Status bar & Spectral Profile
        event_bus.pixel_hovered.connect(self._on_pixel_hovered)
        event_bus.pixel_clicked.connect(self._on_pixel_clicked)

    def open_raster_file(self, file_path: str) -> Optional[RasterLayer]:
        """Open and display a remote sensing image cube (ENVI or GeoTIFF).

        Args:
            file_path: File path to image or header.

        Returns:
            The loaded RasterLayer instance, or None on failure.
        """
        try:
            reader = open_raster(file_path)
            raw_base = os.path.basename(file_path)
            # Prettify Landsat MTL package name
            if "_MTL.txt" in raw_base or "_mtl.txt" in raw_base:
                clean_name = raw_base.replace("_MTL.txt", "").replace("_mtl.txt", "")
            else:
                clean_name = raw_base

            layer_id = f"layer_{len(self._layers) + 1}_{clean_name}"
            meta = reader.metadata
            is_rgb = (meta.default_bands is not None and len(meta.default_bands) == 3) or (meta.bands >= 3)
            layer = RasterLayer(
                layer_id=layer_id,
                name=clean_name,
                file_path=file_path,
                metadata=meta,
                is_visible=True,
                display_mode="rgb" if is_rgb else "grayscale",
            )

            self._readers[layer_id] = reader
            self._layers[layer_id] = layer
            self._active_layer_id = layer_id

            self.dock_layer_manager.add_layer(layer)
            self.dock_data_manager.add_dataset(layer)

            # Auto-display: Prioritize calibrated default RGB (e.g. Landsat True Color B4-B3-B2)
            if meta.default_bands and len(meta.default_bands) == 3:
                r, g, b = meta.default_bands
                self.load_rgb_composition(layer_id, r, g, b)
            elif meta.bands >= 3:
                self.load_rgb_composition(layer_id, 0, 1, 2)
            else:
                self.load_grayscale_band(layer_id, 0)

            event_bus.status_message.emit(tr("status.msg_opened").format(name=layer.name), 3000)
            return layer
        except Exception as e:
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                if "allocate" in str(e).lower() or "memory" in str(e).lower() or isinstance(e, MemoryError):
                    msg = (
                        "打开遥感影像失败：系统连续物理内存不足 (无法分配所需连续内存空间)。\n\n"
                        "原因分析：当前遥感影像尺寸巨大（单波段数千万像元），且当前操作系统可用连续物理内存较低。\n\n"
                        "建议解决方案：\n"
                        "1. 在左侧「图层管理器」中右键移除不需要的旧图层，释放内存空间；\n"
                        "2. 关闭后台高内存占用软件（如浏览器多标签、大型工程软件等）；\n"
                        "3. 调大 Windows 系统的虚拟内存（分页文件 Pagefile）。"
                        if i18n.current_language == "zh"
                        else f"Failed to open image: Insufficient system memory (Out of Memory).\n\n"
                             f"Details: {e}\n\n"
                             f"Suggestions:\n"
                             f"1. Remove unused layers from Layer Manager to free up memory;\n"
                             f"2. Close other memory-intensive applications;\n"
                             f"3. Increase Windows virtual memory (pagefile) size."
                    )
                    QMessageBox.critical(self, tr("dialog.error"), msg)
                else:
                    QMessageBox.critical(self, tr("dialog.error"), f"Failed to open image:\n{e}")
            return None

    def add_derived_layer(
        self,
        name: str,
        data: np.ndarray,
        parent_metadata: Optional[object] = None,
        display_image: Optional[np.ndarray] = None,
    ) -> RasterLayer:
        """Register an in-memory computed array as a new project raster layer."""
        parent_meta = parent_metadata
        if parent_meta is None and self._active_layer_id and self._active_layer_id in self._readers:
            parent_meta = self._readers[self._active_layer_id].metadata

        reader = MemoryRasterReader(data, name=name, parent_metadata=parent_meta)
        layer_id = f"derived_{len(self._layers) + 1}_{name}"
        layer = RasterLayer(
            layer_id=layer_id,
            name=name,
            file_path=f"memory://{name}",
            metadata=reader.metadata,
            is_visible=True,
        )

        if display_image is not None:
            layer.set_thematic_image(display_image)
            if not layer.metadata.raw_header:
                layer.metadata.raw_header = {}
            layer.metadata.raw_header["file type"] = "ENVI Classification"

        self._readers[layer_id] = reader
        self._layers[layer_id] = layer
        self._active_layer_id = layer_id

        self.dock_layer_manager.add_layer(layer)
        self.dock_data_manager.add_dataset(layer)

        # Display immediately in main view
        if display_image is not None:
            self.main_view.display_raster(display_image, reset_view=False)
        elif data.ndim == 3 and data.shape[-1] == 3:
            self.main_view.display_raster(data, reset_view=False)
        else:
            self.main_view.display_raster(reader.read_band(0), reset_view=False)

        event_bus.status_message.emit(f"Created derived layer: {name}", 3000)
        return layer

    @Slot(str, int)
    def load_grayscale_band(self, layer_id: str, band_idx: int) -> None:
        """Load and display a single band in grayscale.

        The resulting stretched uint8 image is cached on the RasterLayer so that
        switching back to this layer later is instantaneous (no disk re-read).
        """
        if layer_id not in self._readers:
            return
        reader = self._readers[layer_id]
        band_data = reader.read_band(band_idx)
        self._active_layer_id = layer_id

        layer = self._layers[layer_id]
        bands_changed = layer.active_bands != (band_idx,) or layer.display_mode != "grayscale"
        layer.display_mode = "grayscale"
        layer.active_bands = (band_idx,)
        if bands_changed:
            layer.invalidate_display_cache()
        self.dock_layer_manager.update_layer_display_mode(layer_id, "grayscale")

        self.main_view.display_raster(band_data, reset_view=False)
        # Cache the rendered result for fast layer switching
        if self.main_view.image_item.image is not None:
            layer.set_display_cache(self.main_view.image_item.image, self.main_view._stretch_mode)
        self.main_view.redraw_rois(getattr(layer, "rois", []))
        event_bus.status_message.emit(f"Displaying: {layer.name} [Band {band_idx + 1}]", 2000)

    @Slot(str, int, int, int)
    def load_rgb_composition(self, layer_id: str, r: int, g: int, b: int) -> None:
        """Load and display a 3-band RGB color composite.

        The resulting stretched uint8 image is cached on the RasterLayer so that
        switching back to this layer later is instantaneous (no disk re-read).
        """
        if layer_id not in self._readers:
            return
        reader = self._readers[layer_id]
        meta = reader.metadata
        h, w = meta.height, meta.width

        try:
            # Preallocate 3D cube directly to avoid holding 3 full separate bands + 1 stacked cube simultaneously
            rgb_cube = np.empty((h, w, 3), dtype=np.float32)
            rgb_cube[..., 0] = reader.read_band(r)
            rgb_cube[..., 1] = reader.read_band(g)
            rgb_cube[..., 2] = reader.read_band(b)
        except (MemoryError, Exception) as e:
            if "allocate" in str(e).lower() or "memory" in str(e).lower() or isinstance(e, MemoryError):
                import gc
                gc.collect()
                msg = (
                    "系统可用物理内存不足以分配 3 个波段的超大 RGB 彩色合成图。\n"
                    "已自动为您降级加载单波段灰度显示以节约连续内存。\n\n"
                    "建议：在左侧图层管理器中移除不需要的旧图层以释放内存。"
                    if i18n.current_language == "zh"
                    else "System RAM is insufficient to allocate full 3-band RGB composite.\n"
                         "Falling back to single band grayscale display to conserve memory.\n"
                         "Tip: Remove unused layers from Layer Manager to free up RAM."
                )
                if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                    QMessageBox.warning(self, tr("dialog.warning"), msg)
                self.load_grayscale_band(layer_id, r)
                return
            raise

        self._active_layer_id = layer_id
        layer = self._layers[layer_id]
        bands_changed = layer.active_bands != (r, g, b) or layer.display_mode != "rgb"
        layer.display_mode = "rgb"
        layer.active_bands = (r, g, b)
        if bands_changed:
            layer.invalidate_display_cache()
        self.dock_layer_manager.update_layer_display_mode(layer_id, "rgb")

        self.main_view.display_raster(rgb_cube, reset_view=False)
        # Cache the rendered result for fast layer switching
        if self.main_view.image_item.image is not None:
            layer.set_display_cache(self.main_view.image_item.image, self.main_view._stretch_mode)
        self.main_view.redraw_rois(getattr(layer, "rois", []))
        event_bus.status_message.emit(
            f"Displaying: {layer.name} [RGB: {r+1}, {g+1}, {b+1}]", 2000
        )

    def _on_active_layer_changed(self, layer_id: str) -> None:
        """Handle layer activation from Layer Manager.

        Uses the per-layer display cache to restore the canvas instantly when
        switching between already-loaded layers, avoiding a full re-read from disk.
        """
        if layer_id not in self._layers:
            return
        self._active_layer_id = layer_id
        layer = self._layers[layer_id]
        if not layer.is_visible:
            return
        # Try fast path: restore from display cache
        cached = layer.get_display_cache(self.main_view._stretch_mode)
        if cached is not None:
            self.main_view.restore_display_cache(cached)
            self.main_view.redraw_rois(getattr(layer, "rois", []))
            return
        # Slow path: re-read from reader (cache miss or stretch mode changed)
        if layer.display_mode == "rgb" and len(layer.active_bands) == 3:
            r, g, b = layer.active_bands
            self.load_rgb_composition(layer_id, r, g, b)
        elif layer.active_bands:
            self.load_grayscale_band(layer_id, layer.active_bands[0])

    def _on_roi_visibility_changed(self, layer_id: str, roi_id: str, is_visible: bool) -> None:
        """Toggle individual ROI visibility from Layer Manager."""
        if layer_id in self._layers:
            layer = self._layers[layer_id]
            for roi in getattr(layer, "rois", []):
                if roi.roi_id == roi_id:
                    roi.is_visible = is_visible
                    break
            if self._active_layer_id == layer_id:
                self.main_view.redraw_rois(getattr(layer, "rois", []))

    def _on_roi_removed(self, layer_id: str, roi_id: str) -> None:
        """Delete ROI from Layer Manager."""
        if layer_id in self._layers:
            layer = self._layers[layer_id]
            if hasattr(layer, "rois"):
                layer.rois = [r for r in layer.rois if r.roi_id != roi_id]
            if self._active_layer_id == layer_id:
                self.main_view.redraw_rois(getattr(layer, "rois", []))
            if hasattr(self, "_roi_dialog") and self._roi_dialog is not None and self._roi_dialog.isVisible():
                if getattr(self._roi_dialog, "layer", None) == layer:
                    self._roi_dialog._refresh_table()

    def _on_layer_visibility_changed(self, layer_id: str, is_visible: bool) -> None:
        """Toggle layer visibility."""
        if layer_id in self._layers:
            self._layers[layer_id].is_visible = is_visible
            if not is_visible and self._active_layer_id == layer_id:
                # Clear canvas if active layer hidden
                self.main_view.image_item.clear()
                self.main_view.overview_img.clear()
            elif is_visible and self._active_layer_id == layer_id:
                layer = self._layers[layer_id]
                if layer.display_mode == "rgb" and len(layer.active_bands) == 3:
                    self.load_rgb_composition(layer_id, *layer.active_bands)
                else:
                    self.load_grayscale_band(layer_id, layer.active_bands[0])

    def _remove_active_layer(self) -> None:
        """Remove active layer from project."""
        if self._active_layer_id:
            self._on_layer_removed(self._active_layer_id)

    @Slot(str)
    def _on_layer_removed(self, layer_id: str) -> None:
        """Handle layer removal and synchronize docks and viewport."""
        if layer_id in self._readers:
            self._readers[layer_id].close()
            del self._readers[layer_id]
        if layer_id in self._layers:
            self._layers[layer_id].invalidate_display_cache()
            del self._layers[layer_id]

        import gc
        gc.collect()

        # Sync docks
        self.dock_layer_manager.remove_layer_by_id(layer_id)
        self.dock_data_manager.remove_dataset(layer_id)

        # If deleted layer was the active layer, switch or clear canvas
        if self._active_layer_id == layer_id:
            if self._layers:
                next_id = next(iter(self._layers))
                self._active_layer_id = next_id
                layer = self._layers[next_id]
                if layer.is_visible:
                    if layer.display_mode == "rgb" and len(layer.active_bands) == 3:
                        self.load_rgb_composition(next_id, *layer.active_bands)
                    else:
                        self.load_grayscale_band(next_id, layer.active_bands[0])
                else:
                    self.main_view.clear()
            else:
                self._active_layer_id = None
                self.main_view.clear()
                self.dock_spectral_profile.clear_spectrum()
                self.status_bar.clear()
        event_bus.status_message.emit(tr("status.msg_layer_removed"), 2000)

    @Slot(int, int)
    def _on_pixel_hovered(self, x: int, y: int) -> None:
        """Handle cursor movement over raster coordinates with 16ms debounce."""
        self._pending_hover_coords = (x, y)
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _do_pixel_hover(self) -> None:
        """Execute debounced hover processing: coordinate projection and spectral sampling."""
        if not self._pending_hover_coords:
            return
        x, y = self._pending_hover_coords
        if not self._active_layer_id or self._active_layer_id not in self._readers:
            return

        reader = self._readers[self._active_layer_id]
        try:
            geo_x, geo_y = reader.pixel_to_geo(x, y)
            crs_str = reader.metadata.crs if hasattr(reader, "metadata") and reader.metadata else None
            self.status_bar.update_geo_coords(geo_x, geo_y, crs_str=crs_str)
        except Exception:
            pass

        # Read active pixel values
        try:
            profile = reader.read_pixel_profile(x, y)
            self.status_bar.update_pixel_values(profile[:3].tolist())
        except Exception:
            pass

    @Slot(int, int)
    def _on_pixel_clicked(self, x: int, y: int) -> None:
        """Handle mouse click / probe on raster canvas."""
        if not self._active_layer_id or self._active_layer_id not in self._readers:
            return

        reader = self._readers[self._active_layer_id]
        try:
            profile = reader.read_pixel_profile(x, y)
            meta = reader.metadata
            wavelengths = np.array([
                b.wavelength for b in meta.band_details if b.wavelength is not None
            ], dtype=np.float32)

            if len(wavelengths) != len(profile):
                wavelengths = None

            self.dock_spectral_profile.set_spectrum(
                values=profile,
                wavelengths=wavelengths,
                x=x,
                y=y,
            )
            event_bus.status_message.emit(
                f"Probed Pixel ({x}, {y}) | Spectrum Loaded", 2000
            )
        except Exception as e:
            event_bus.status_message.emit(f"Error reading spectrum: {e}", 3000)

    def _on_toolbox_tool_selected(self, tool_name: str) -> None:
        """Launch corresponding algorithm dialog when clicked in Toolbox."""
        tl = tool_name.lower()
        if "band_math" in tl or "math" in tl:
            self.show_band_math_dialog()
        elif "export" in tl or "save" in tl:
            self.show_export_dialog()
        elif "pansharpen" in tl or "sharpen" in tl or "fusion" in tl:
            self.show_pansharpen_dialog()
        elif "radiometry" in tl or "calibration" in tl or "dos" in tl:
            self.show_radiometry_dialog()
        elif any(k in tl for k in ("ndvi", "ndwi", "evi", "savi", "nbr", "index", "indices")):
            self.show_indices_dialog()
        elif "mnf" in tl:
            self.show_mnf_dialog()
        elif "ica" in tl or "independent" in tl:
            self.show_ica_dialog()
        elif "pca" in tl or "principal" in tl:
            self.show_pca_dialog()
        elif "sam" in tl:
            self.show_sam_dialog()
        elif "sid" in tl or "divergence" in tl:
            self.show_sid_dialog()
        elif "sff" in tl or "feature fitting" in tl:
            self.show_sff_dialog()
        elif "maxlik" in tl or "maximum" in tl:
            self.show_maxlik_dialog()
        elif "svm" in tl or "support vector" in tl:
            self.show_svm_dialog()
        elif any(k in tl for k in ("kmeans", "isodata", "classification")):
            self.show_classification_dialog()
        elif "roi" in tl or "region" in tl:
            self.show_roi_dialog()
        elif "stat" in tl:
            self.show_stats_dialog()
        elif "stack" in tl:
            self.show_layer_stacking_dialog()
        elif "mosaic" in tl:
            self.show_mosaic_dialog()
        elif "resize" in tl or "subset" in tl:
            self.show_resize_dialog()
        elif "color" in tl or "hsv" in tl:
            self.show_color_transform_dialog()
        elif "continuum" in tl or "convex" in tl:
            self.show_continuum_removal_dialog()
        elif "accuracy" in tl or "confusion" in tl or "matrix" in tl:
            self.show_accuracy_assessment_dialog()

    def get_available_layers(self) -> Dict[str, Tuple[RasterLayer, BaseRasterReader]]:
        """Return dictionary of loaded layers and their readers."""
        return {
            lid: (self._layers[lid], self._readers[lid])
            for lid in self._layers
            if lid in self._readers
        }

    def show_export_dialog(self, layer_id: Optional[str] = None) -> None:
        """Open Export Raster dialog."""
        target_id = layer_id or self._active_layer_id
        if not target_id or target_id not in self._layers or target_id not in self._readers:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return

        available = self.get_available_layers()
        dlg = ExportRasterDialog(
            layer=self._layers[target_id],
            reader=self._readers[target_id],
            available_layers=available,
            parent=self,
        )
        dlg.export_completed.connect(self._on_raster_exported)
        dlg.exec()

    def _on_raster_exported(self, file_path: str) -> None:
        """Handle completion of raster export by loading exported file."""
        if file_path and os.path.exists(file_path):
            self.open_raster_file(file_path)

    def show_pansharpen_dialog(self) -> None:
        """Open Pan-Sharpening image fusion dialog."""
        available = self.get_available_layers()
        if not available:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return

        dlg = PanSharpenDialog(
            available_layers=available,
            active_layer_id=self._active_layer_id,
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_radiometry_dialog(self) -> None:
        """Open Radiometric Calibration and Atmospheric Correction dialog."""
        available = self.get_available_layers()
        if not available:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return

        dlg = RadiometryDialog(
            available_layers=available,
            active_layer_id=self._active_layer_id,
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_band_math_dialog(self) -> None:
        """Open Band Math dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = BandMathDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_indices_dialog(self) -> None:
        """Open Spectral Indices dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = IndicesDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_pca_dialog(self) -> None:
        """Open PCA dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = PCADialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            mode="pca",
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_mnf_dialog(self) -> None:
        """Open Minimum Noise Fraction (MNF) transform dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = PCADialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            mode="mnf",
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_ica_dialog(self) -> None:
        """Open Independent Component Analysis (ICA) transform dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = PCADialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            mode="ica",
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_sam_dialog(self) -> None:
        """Open Spectral Angle Mapper (SAM) supervised classification dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = SAMDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_classification_dialog(self) -> None:
        """Open Classification dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = ClassificationDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_maxlik_dialog(self) -> None:
        """Open Maximum Likelihood Classification (MLC) supervised dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = MaximumLikelihoodDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.open_roi_tool_requested.connect(self.show_roi_dialog)
        dlg.exec()

    def show_sid_dialog(self) -> None:
        """Open Spectral Information Divergence (SID) supervised classification dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = SIDDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_svm_dialog(self) -> None:
        """Open Support Vector Machine (SVM) supervised classification dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = SVMDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_roi_dialog(self, target_layer_id: Optional[str] = None) -> None:
        """Open ROI Tool dialog with interactive polygon canvas drawing (modeless)."""
        layer_id = target_layer_id if (target_layer_id and target_layer_id in self._layers) else self._active_layer_id
        if not layer_id or layer_id not in self._layers:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        if hasattr(self, "_roi_dialog") and self._roi_dialog is not None and self._roi_dialog.isVisible():
            if getattr(self._roi_dialog, "layer", None) != self._layers[layer_id]:
                self._roi_dialog.close()
            else:
                self._roi_dialog.raise_()
                self._roi_dialog.activateWindow()
                return

        layer = self._layers[layer_id]
        reader = self._readers[layer_id]
        self._roi_dialog = ROIToolDialog(
            layer=layer,
            reader=reader,
            main_view=self.main_view,
            parent=self,
        )
        self._roi_dialog.plot_mean_spectrum_requested.connect(self._on_roi_plot_mean_spectrum)
        self._roi_dialog.mask_generated.connect(self.add_derived_layer)
        self._roi_dialog.roi_updated.connect(lambda: self.dock_layer_manager.sync_layer_rois(layer))
        self._roi_dialog.show()

    def _on_roi_plot_mean_spectrum(
        self,
        values: np.ndarray,
        wavelengths: Optional[np.ndarray],
        name: str,
        color: str,
    ) -> None:
        """Display ROI mean spectrum in Spectral Profile dock and bring dock to front."""
        self.dock_spectral_profile.show()
        self.dock_spectral_profile.raise_()
        self.dock_spectral_profile.add_spectrum_overlay(values, wavelengths, name, color)

    def show_stats_dialog(self) -> None:
        """Open Quick Statistics dialog for active layer."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = QuickStatsDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.exec()

    def _save_view_image(self) -> None:
        """Export the currently displayed stretched image to PNG/JPEG/BMP."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        layer_name = self._layers[self._active_layer_id].name
        clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in layer_name)
        default_path = f"{clean_name}_view.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            tr("menu.save_view_image"),
            default_path,
            "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp)",
        )
        if not file_path:
            return
        ok = self.main_view.save_view_image(file_path)
        if ok:
            event_bus.status_message.emit(f"Viewport saved to: {file_path}", 4000)
            QMessageBox.information(self, tr("dialog.export.success_title"), f"Image saved successfully to:\n{file_path}")
        else:
            QMessageBox.critical(self, tr("dialog.export.err_title"), "Failed to save viewport image.")

    def show_layer_stacking_dialog(self) -> None:
        """Open Layer Stacking dialog to combine bands from multiple datasets."""
        layers = self.get_available_layers()
        dlg = LayerStackingDialog(layers, parent=self)
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_resize_dialog(self) -> None:
        """Open Resize Data / Subsetting dialog for active layer."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = ResizeDataDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_color_transform_dialog(self) -> None:
        """Open Color Space Transform (RGB-HSV) dialog for active layer."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = ColorTransformDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_sff_dialog(self) -> None:
        """Open Spectral Feature Fitting dialog for active layer."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = SFFDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_mosaic_dialog(self) -> None:
        """Open Mosaicking dialog to combine two or more open rasters."""
        layers = self.get_available_layers()
        if not layers:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = MosaicDialog(layers, parent=self)
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_continuum_removal_dialog(self) -> None:
        """Open Continuum Removal dialog for active layer."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = ContinuumRemovalDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.result_generated.connect(self.add_derived_layer)
        dlg.exec()

    def show_accuracy_assessment_dialog(self) -> None:
        """Open Confusion Matrix & Accuracy Assessment dialog."""
        layers = self.get_available_layers()
        if len(layers) < 1:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = AccuracyAssessmentDialog(
            layers=layers,
            active_layer_id=self._active_layer_id,
            parent=self,
        )
        dlg.exec()

    def show_synthetic_generator_dialog(self) -> None:
        """Open Synthetic Benchmark Generator dialog."""
        dlg = SyntheticDataDialog(self)
        dlg.dataset_generated.connect(self.open_raster_file)
        dlg.exec()

    def retranslate_ui(self) -> None:
        """Update all main window texts when language changes."""
        self.setWindowTitle(tr("app.title"))

        # Menus
        self.menu_file.setTitle(tr("menu.file"))
        self.menu_view.setTitle(tr("menu.view"))
        self.menu_layer.setTitle(tr("menu.layer"))
        self.menu_display.setTitle(tr("menu.display"))
        self.menu_tools.setTitle(tr("menu.tools"))
        self.menu_window.setTitle(tr("menu.window"))
        self.menu_language.setTitle(tr("menu.language"))
        self.menu_help.setTitle(tr("menu.help"))

        # Actions
        self.act_open.setText(tr("action.open"))
        self.act_export_raster.setText(tr("menu.export_raster"))
        self.act_save_view_image.setText(tr("menu.save_view_image"))
        self.act_gen_data.setText(tr("action.gen_data"))
        self.act_exit.setText(tr("action.exit"))
        self.act_probe.setText(tr("action.probe"))
        self.act_pan.setText(tr("action.pan"))
        self.act_zoom_in.setText(tr("action.zoom_in"))
        self.act_zoom_out.setText(tr("action.zoom_out"))
        self.act_zoom_fit.setText(tr("action.fit"))
        self.act_toggle_overview.setText(tr("main_view.overview_toggle"))
        self.act_layer_props.setText(tr("action.layer_props"))
        self.act_remove_layer.setText(tr("action.remove_layer"))
        self.act_reset_layout.setText(tr("action.reset_layout"))
        self.act_shortcuts.setText(tr("action.shortcuts"))
        self.act_about.setText(tr("action.about"))

        # Tools Menu Actions
        self.act_roi.setText(tr("action.roi"))
        self.act_band_math.setText(tr("action.band_math"))
        self.act_indices.setText(tr("action.indices"))
        self.act_pca.setText(tr("action.pca"))
        self.act_mnf.setText(tr("action.mnf"))
        self.act_sam.setText(tr("action.sam"))
        self.act_classification.setText(tr("action.classification"))
        self.act_maxlik.setText(tr("action.maxlik"))
        self.act_pansharpen.setText(tr("menu.pansharpen"))
        self.act_radiometry.setText(tr("menu.radiometry"))
        self.act_stats.setText(tr("action.stats"))
        self.act_stacking.setText(tr("menu.stacking"))
        self.act_mosaic.setText(tr("menu.mosaic"))
        self.act_resize.setText(tr("menu.resize"))
        self.act_color.setText(tr("menu.color"))
        self.act_continuum.setText(tr("menu.continuum"))
        self.act_accuracy.setText(tr("menu.accuracy"))

        # Display Menu Stretch Actions
        self.act_stretch_lin2.setText(tr("stretch.linear2"))
        self.act_stretch_lin5.setText(tr("stretch.linear5"))
        self.act_stretch_eq.setText(tr("stretch.equalize"))
        self.act_stretch_gauss.setText(tr("stretch.gaussian"))

        # Stretch Selector
        self.lbl_stretch.setText(tr("stretch.label"))
        current_idx = self.cb_stretch.currentIndex()
        self.cb_stretch.blockSignals(True)
        self.cb_stretch.clear()
        self.cb_stretch.addItems([
            tr("stretch.linear2"),
            tr("stretch.linear5"),
            tr("stretch.equalize"),
            tr("stretch.gaussian"),
            tr("stretch.none"),
        ])
        if current_idx >= 0:
            self.cb_stretch.setCurrentIndex(current_idx)
        self.cb_stretch.blockSignals(False)

        # Language action check states
        self.act_lang_en.setChecked(i18n.current_language == "en")
        self.act_lang_zh.setChecked(i18n.current_language == "zh")

        event_bus.status_message.emit(tr("app.initialized"), 3000)

    def _set_language(self, lang: str) -> None:
        """Handle language change from menu."""
        i18n.set_language(lang)
        settings = QSettings("OpenENVI", "OpenENVI")
        settings.setValue("language", lang)
        event_bus.language_changed.emit(lang)

    def closeEvent(self, event) -> None:
        """Save window layout, geometry, language, and stretch enhancement upon exit."""
        settings = QSettings("OpenENVI", "OpenENVI")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("language", i18n.current_language)
        settings.setValue("stretch_mode", self.cb_stretch.currentText())
        super().closeEvent(event)

    def _on_stretch_changed(self, mode: str) -> None:
        """Handle contrast stretch mode change.

        Updates the stretch mode on the view, invalidates the display cache on
        all loaded layers so layer switching reflects the new stretch mode, then
        re-loads the active layer's currently selected bands from the reader.
        """
        self.main_view._stretch_mode = mode
        for layer in self._layers.values():
            layer.invalidate_display_cache()

        if not self._active_layer_id or self._active_layer_id not in self._layers:
            return
        layer = self._layers[self._active_layer_id]
        if not layer.is_visible:
            return
        if layer.display_mode == "rgb" and len(layer.active_bands) == 3:
            r, g, b = layer.active_bands
            self.load_rgb_composition(self._active_layer_id, r, g, b)
        elif layer.active_bands:
            self.load_grayscale_band(self._active_layer_id, layer.active_bands[0])

    def _on_open_file_dialog(self) -> None:
        """Open raster file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("dialog.open_title"),
            "",
            tr("dialog.open_filter"),
        )
        if file_path:
            self.open_raster_file(file_path)

    def _reset_dock_layout(self) -> None:
        """Reset docks to their default arrangement."""
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_layer_manager)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_data_manager)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_toolbox)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_spectral_profile)
        self.dock_layer_manager.show()
        self.dock_data_manager.show()
        self.dock_toolbox.show()
        self.dock_spectral_profile.show()

    def _on_shortcuts(self) -> None:
        """Show User Guide & Keyboard Shortcuts Dialog."""
        QMessageBox.information(
            self,
            tr("app.shortcuts_title"),
            tr("app.shortcuts_desc"),
        )

    def _on_about(self) -> None:
        """Show About Dialog."""
        QMessageBox.about(
            self,
            tr("app.about_title"),
            tr("app.about_desc"),
        )
