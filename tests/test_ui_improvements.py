"""Unit tests for RGB Panel, Z-Profile Sorting, Ready Translation, and Overview Toggle."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.main import create_app
from core.i18n import i18n, tr
from core.models import BandInfo, RasterLayer, RasterMetadata
from ui.data_manager import DataManagerDock
from ui.spectral_profile import SpectralProfileDock
from ui.status_bar import OpenENVIStatusBar
from ui.main_view import MainViewWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_status_bar_ready_translation(qapp):
    """Verify status bar initializes with translated ready string and handles language changes."""
    i18n.set_language("zh")
    assert tr("app.ready") == "就绪"

    status_bar = OpenENVIStatusBar()
    assert status_bar._lbl_status.text() in ("就绪", tr("app.initialized"))

    # Set temporary message and verify timeout callback restores translated string
    status_bar.set_status("测试消息", timeout=10)
    assert status_bar._lbl_status.text() == "测试消息"

    # Process events to trigger singleShot timeout
    from PySide6.QtCore import QCoreApplication
    import time
    time.sleep(0.05)
    QCoreApplication.processEvents()
    assert status_bar._lbl_status.text() in ("就绪", tr("app.initialized"))

    # Switch back to English
    i18n.set_language("en")
    assert tr("app.ready") == "Ready"


def test_spectral_profile_wavelength_sorting_and_si_prefix(qapp):
    """Verify that spectral profile disables SI prefix and sorts out-of-order wavelengths."""
    dock = SpectralProfileDock()
    axis = dock.plot_widget.getPlotItem().getAxis("bottom")
    assert axis.autoSIPrefix is False

    # Out-of-order wavelengths: e.g. Landsat Band 7 (2201 nm), Band 9 Cirrus (1373 nm), Band 10 TIRS (10895 nm)
    wavelengths = np.array([443.0, 655.0, 2201.0, 1373.0, 10895.0])
    values = np.array([100.0, 250.0, 180.0, 40.0, 12000.0])

    dock.set_spectrum(values, wavelengths, x=10, y=20)

    # Check plotted curve data
    x_data, y_data = dock._curve.getData()
    # x_data must be strictly monotonically increasing (no backwards zig-zag!)
    assert np.all(np.diff(x_data) > 0)
    assert list(x_data) == [443.0, 655.0, 1373.0, 2201.0, 10895.0]

    # Switch to Band Number mode
    dock._combo_x_axis.setCurrentIndex(1)  # band
    x_band, y_band = dock._curve.getData()
    assert list(x_band) == [1, 2, 3, 4, 5]
    assert list(y_band) == list(values)


def test_data_manager_rgb_slots_and_assignment(qapp):
    """Verify DataManagerDock RGB channel slot interactions, auto-advance, and loading."""
    dock = DataManagerDock()
    dock.show()

    # Create dummy 10-band layer (e.g. Landsat-like)
    bands = [
        BandInfo(index=i, name=f"Band {i+1}", wavelength=400.0 + i * 100.0, wavelength_unit="nm")
        for i in range(10)
    ]
    meta = RasterMetadata(width=100, height=100, bands=10, dtype="float32", band_details=bands)
    layer = RasterLayer(layer_id="test_l8", name="LC08_Test", file_path="/dummy/mtl.txt", metadata=meta)

    dock.add_dataset(layer)

    # Check auto-defaulting for 10-band dataset (Landsat RGB = bands 4, 3, 2 -> indices 3, 2, 1)
    assert dock._rgb_bands == [3, 2, 1]
    assert "Band 4" in dock.btn_slot_r.text()
    assert "Band 3" in dock.btn_slot_g.text()
    assert "Band 2" in dock.btn_slot_b.text()

    # Switch to RGB mode
    dock.rb_rgb.setChecked(True)
    assert not dock.rgb_box.isHidden()
    assert dock._active_rgb_slot == 0  # Red slot active

    # Manual slot selection
    dock.btn_slot_g.click()
    assert dock._active_rgb_slot == 1  # Green slot active

    # Emulate clicking a band in the tree
    band_item = dock.tree.topLevelItem(0).child(5)  # Band 6 (index 5)
    dock._on_item_clicked(band_item, 0)

    # Slot G should now be Band 6, and active slot advances to B (slot 2)
    assert dock._rgb_bands[1] == 5
    assert "Band 6" in dock.btn_slot_g.text()
    assert dock._active_rgb_slot == 2


def test_main_view_overview_collapse_and_toggle(qapp):
    """Verify overview widget minimizes, expands, and respects visibility toggles."""
    view = MainViewWidget()
    view.show()
    assert not view.overview_frame.isHidden()
    assert view._overview_collapsed is False
    assert view.overview_frame.height() == 150

    # Collapse overview
    view.toggle_overview_collapsed()
    assert view._overview_collapsed is True
    assert view.overview_glw.isHidden() is True
    assert view.overview_frame.height() == 24

    # Expand overview
    view.toggle_overview_collapsed()
    assert view._overview_collapsed is False
    assert view.overview_glw.isHidden() is False
    assert view.overview_frame.height() == 150

    # Hide overview completely
    view.set_overview_visible(False)
    assert view.overview_frame.isHidden() is True
