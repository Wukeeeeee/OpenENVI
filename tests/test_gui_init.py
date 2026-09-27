"""Headless Automated Initialization Tests for OpenENVI.

Tests application startup, dock creation, event bus signal propagation,
and core dataclasses under offscreen Qt platform mode.
"""

import os
import sys

# Ensure offscreen rendering for headless CI / automated testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.main import create_app
from app.main_window import OpenENVIMainWindow
from core.events import AppEventBus, event_bus
from core.models import BandInfo, RasterLayer, RasterMetadata
from ui.data_manager import DataManagerDock
from ui.layer_manager import LayerManagerDock
from ui.main_view import MainViewWidget
from ui.spectral_profile import SpectralProfileDock
from ui.status_bar import OpenENVIStatusBar
from ui.toolbox import ToolboxDock


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single persistent QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_core_models():
    """Verify that core dataclasses instantiate correctly."""
    band = BandInfo(index=0, name="Red", wavelength=665.0, wavelength_unit="nm")
    assert band.display_name() == "Band 1 (665.00 nm)"

    meta = RasterMetadata(
        width=512,
        height=512,
        bands=10,
        dtype="float32",
        band_details=[band],
    )
    assert meta.width == 512
    assert meta.bands == 10
    assert len(meta.band_details) == 1

    layer = RasterLayer(
        layer_id="layer_001",
        name="test_raster",
        file_path="/path/to/test.hdr",
        metadata=meta,
    )
    assert layer.layer_id == "layer_001"
    assert layer.is_visible is True
    assert layer.display_mode == "grayscale"


def test_event_bus():
    """Verify event bus signal emission and slot reception."""
    bus = AppEventBus()

    hover_coords = []
    click_coords = []
    layer_names = []
    stretches = []

    bus.pixel_hovered.connect(lambda x, y: hover_coords.append((x, y)))
    bus.pixel_clicked.connect(lambda x, y: click_coords.append((x, y)))
    bus.layer_changed.connect(lambda name: layer_names.append(name))
    bus.stretch_mode_changed.connect(lambda mode: stretches.append(mode))

    bus.pixel_hovered.emit(10, 20)
    bus.pixel_clicked.emit(45, 67)
    bus.layer_changed.emit("sample_layer")
    bus.stretch_mode_changed.emit("Linear 2%")

    assert hover_coords == [(10, 20)]
    assert click_coords == [(45, 67)]
    assert layer_names == ["sample_layer"]
    assert stretches == ["Linear 2%"]


def test_main_window_init(qapp):
    """Verify that OpenENVIMainWindow instantiates all 5 core panels without error."""
    window = OpenENVIMainWindow()

    # Verify Central Widget
    assert isinstance(window.main_view, MainViewWidget)
    assert window.centralWidget() == window.main_view

    # Verify Status Bar
    assert isinstance(window.status_bar, OpenENVIStatusBar)
    assert window.statusBar() == window.status_bar

    # Verify All 4 Dock Widgets
    assert isinstance(window.dock_layer_manager, LayerManagerDock)
    assert isinstance(window.dock_data_manager, DataManagerDock)
    assert isinstance(window.dock_toolbox, ToolboxDock)
    assert isinstance(window.dock_spectral_profile, SpectralProfileDock)

    assert window.dock_layer_manager.windowTitle() == "Layer Manager"
    assert window.dock_data_manager.windowTitle() == "Data Manager"
    assert window.dock_toolbox.windowTitle() == "Toolbox"
    assert window.dock_spectral_profile.windowTitle() == "Spectral Profile (Z-Profile)"

    window.close()


def test_main_view_canvas(qapp):
    """Verify raster display canvas and zoom controls."""
    main_view = MainViewWidget()
    test_data = np.random.rand(100, 100).astype(np.float32)
    main_view.display_raster(test_data)

    assert main_view._raster_w == 100
    assert main_view._raster_h == 100

    # Test zoom methods execution
    main_view.zoom_in()
    main_view.zoom_out()
    main_view.zoom_fit()


def test_spectral_profile_dock(qapp):
    """Verify Spectral Profile curve setting and clearing."""
    dock = SpectralProfileDock()
    values = np.array([0.1, 0.25, 0.45, 0.8, 0.6, 0.3], dtype=np.float32)
    wavelengths = np.array([450.0, 520.0, 660.0, 840.0, 1600.0, 2200.0])

    dock.set_spectrum(values, wavelengths, x=50, y=50)
    assert "Position: (50, 50)" in dock._lbl_info.text()
    assert "Bands: 6" in dock._lbl_info.text()

    dock.clear_spectrum()
    assert "Cleared" in dock._lbl_info.text()


def test_create_app():
    """Verify create_app helper initializes QApplication and Main Window."""
    app, window = create_app()
    assert app is not None
    assert window is not None
    assert isinstance(window, OpenENVIMainWindow)
    window.close()
