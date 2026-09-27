"""OpenENVI Main Window.

Assembles the dockable workspace modeled after ENVI Classic and ENVI 5.x,
integrating Layer Manager, Data Manager, Toolbox, Spectral Profile, Main View,
interactive algorithm dialogs, and real-time cursor/spectral probing.
"""

import os
from typing import Dict, List, Optional
import numpy as np
from PySide6.QtCore import Qt, Slot
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
from ui.dialogs.band_math_dialog import BandMathDialog
from ui.dialogs.classification_dialog import ClassificationDialog
from ui.dialogs.indices_dialog import IndicesDialog
from ui.dialogs.pca_dialog import PCADialog
from ui.dialogs.roi_dialog import ROIToolDialog
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

        self.act_classification = QAction("Image Classification...", self)
        self.act_classification.triggered.connect(self.show_classification_dialog)

        self.act_roi = QAction("Region of Interest (ROI) Tool...", self)
        self.act_roi.triggered.connect(self.show_roi_dialog)

        # Window & Help Actions
        self.act_reset_layout = QAction("Reset Dock Layout", self)
        self.act_reset_layout.triggered.connect(self._reset_dock_layout)

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
        self.menu_tools.addAction(self.act_classification)

        # Window Menu
        self.menu_window = mb.addMenu("&Window")
        self.menu_window.addAction(self.act_reset_layout)

        # Language Menu
        self.menu_language = mb.addMenu("&Language")
        self.menu_language.addAction(self.act_lang_en)
        self.menu_language.addAction(self.act_lang_zh)

        # Help Menu
        self.menu_help = mb.addMenu("&Help")
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

        # Layer Manager interactions
        self.dock_layer_manager.layer_visibility_changed.connect(self._on_layer_visibility_changed)
        self.dock_layer_manager.layer_removed.connect(self._on_layer_removed)
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
            layer_id = f"layer_{len(self._layers) + 1}_{os.path.basename(file_path)}"
            layer = RasterLayer(
                layer_id=layer_id,
                name=os.path.basename(file_path),
                file_path=file_path,
                metadata=reader.metadata,
                is_visible=True,
            )

            self._readers[layer_id] = reader
            self._layers[layer_id] = layer
            self._active_layer_id = layer_id

            self.dock_layer_manager.add_layer(layer)
            self.dock_data_manager.add_dataset(layer)

            # Auto-display: RGB if >= 3 bands, else Band 0 Grayscale
            if reader.metadata.bands >= 3:
                self.load_rgb_composition(layer_id, 0, 1, 2)
            else:
                self.load_grayscale_band(layer_id, 0)

            event_bus.status_message.emit(f"Opened: {layer.name}", 3000)
            return layer
        except Exception as e:
            QMessageBox.critical(self, "Open Image Error", f"Failed to open image:\n{e}")
            return None

    def add_derived_layer(self, name: str, data: np.ndarray) -> RasterLayer:
        """Register an in-memory computed array as a new project raster layer."""
        parent_meta = None
        if self._active_layer_id and self._active_layer_id in self._readers:
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

        self._readers[layer_id] = reader
        self._layers[layer_id] = layer
        self._active_layer_id = layer_id

        self.dock_layer_manager.add_layer(layer)
        self.dock_data_manager.add_dataset(layer)

        # Display immediately in main view
        if data.ndim == 3 and data.shape[-1] == 3:
            self.main_view.display_raster(data, reset_view=False)
        elif data.ndim == 2:
            self.main_view.display_raster(data, reset_view=False)
        else:
            self.main_view.display_raster(data[0], reset_view=False)

        event_bus.status_message.emit(f"Created derived layer: {name}", 3000)
        return layer

    @Slot(str, int)
    def load_grayscale_band(self, layer_id: str, band_idx: int) -> None:
        """Load and display a single band in grayscale."""
        if layer_id not in self._readers:
            return
        reader = self._readers[layer_id]
        band_data = reader.read_band(band_idx)
        self._active_layer_id = layer_id

        layer = self._layers[layer_id]
        layer.display_mode = "grayscale"
        layer.active_bands = (band_idx,)

        self.main_view.display_raster(band_data, reset_view=False)
        event_bus.status_message.emit(f"Displaying: {layer.name} [Band {band_idx + 1}]", 2000)

    @Slot(str, int, int, int)
    def load_rgb_composition(self, layer_id: str, r: int, g: int, b: int) -> None:
        """Load and display a 3-band RGB color composite."""
        if layer_id not in self._readers:
            return
        reader = self._readers[layer_id]
        r_band = reader.read_band(r)
        g_band = reader.read_band(g)
        b_band = reader.read_band(b)
        rgb_cube = np.stack([r_band, g_band, b_band], axis=-1)

        self._active_layer_id = layer_id
        layer = self._layers[layer_id]
        layer.display_mode = "rgb"
        layer.active_bands = (r, g, b)

        self.main_view.display_raster(rgb_cube, reset_view=False)
        event_bus.status_message.emit(
            f"Displaying: {layer.name} [RGB: {r+1}, {g+1}, {b+1}]", 2000
        )

    def _on_active_layer_changed(self, layer_id: str) -> None:
        """Handle layer activation from Layer Manager."""
        if layer_id in self._layers:
            self._active_layer_id = layer_id

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
            del self._layers[layer_id]

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
        event_bus.status_message.emit("Layer removed", 2000)

    @Slot(int, int)
    def _on_pixel_hovered(self, x: int, y: int) -> None:
        """Handle cursor movement over raster coordinates."""
        if not self._active_layer_id or self._active_layer_id not in self._readers:
            return

        reader = self._readers[self._active_layer_id]
        geo_x, geo_y = reader.pixel_to_geo(x, y)
        self.status_bar.update_geo_coords(geo_y, geo_x)

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
        elif any(k in tl for k in ("ndvi", "ndwi", "evi", "savi", "nbr", "index", "indices")):
            self.show_indices_dialog()
        elif "pca" in tl or "mnf" in tl or "principal" in tl:
            self.show_pca_dialog()
        elif any(k in tl for k in ("kmeans", "isodata", "classification", "sam", "maxlik")):
            self.show_classification_dialog()
        elif "roi" in tl or "region" in tl:
            self.show_roi_dialog()

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

    def show_roi_dialog(self) -> None:
        """Open ROI Tool dialog."""
        if not self._active_layer_id:
            QMessageBox.information(self, tr("dialog.no_active_title"), tr("dialog.no_active_layer"))
            return
        dlg = ROIToolDialog(
            layer=self._layers[self._active_layer_id],
            reader=self._readers[self._active_layer_id],
            parent=self,
        )
        dlg.plot_mean_spectrum_requested.connect(self.dock_spectral_profile.add_spectrum_overlay)
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
        self.act_exit.setText(tr("action.exit"))
        self.act_probe.setText(tr("action.probe"))
        self.act_pan.setText(tr("action.pan"))
        self.act_zoom_in.setText(tr("action.zoom_in"))
        self.act_zoom_out.setText(tr("action.zoom_out"))
        self.act_zoom_fit.setText(tr("action.fit"))
        self.act_layer_props.setText(tr("action.layer_props"))
        self.act_remove_layer.setText(tr("action.remove_layer"))
        self.act_reset_layout.setText(tr("action.reset_layout"))
        self.act_about.setText(tr("action.about"))

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
        event_bus.language_changed.emit(lang)

    def _on_stretch_changed(self, mode: str) -> None:
        """Handle contrast stretch mode change."""
        event_bus.stretch_mode_changed.emit(mode)

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

    def _on_about(self) -> None:
        """Show About Dialog."""
        QMessageBox.about(
            self,
            tr("app.about_title"),
            tr("app.about_desc"),
        )
