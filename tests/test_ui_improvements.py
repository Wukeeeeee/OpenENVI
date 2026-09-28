"""Unit tests for RGB Panel, Z-Profile Sorting, Ready Translation, and Overview Toggle."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.main import create_app
from core.i18n import i18n, tr
from core.io.memory import MemoryRasterReader
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


def test_hover_debounce_and_display_cache_invalidation(qapp):
    """Verify hover debounce timer behavior, display cache invalidation on stretch changes, and layer cleanup."""
    from app.main_window import OpenENVIMainWindow
    from core.io.memory import MemoryRasterReader
    from core.events import event_bus

    window = OpenENVIMainWindow()

    # Create two test raster layers with memory readers
    data1 = np.ones((50, 50, 4), dtype=np.float32) * 10.0
    data2 = np.ones((50, 50, 4), dtype=np.float32) * 20.0

    layer1 = window.add_derived_layer("Layer1", data1)
    layer2 = window.add_derived_layer("Layer2", data2)

    # Set display cache on both layers
    cache1 = np.zeros((50, 50, 3), dtype=np.uint8)
    cache2 = np.zeros((50, 50, 3), dtype=np.uint8)
    layer1.set_display_cache(cache1, "Linear 2%")
    layer2.set_display_cache(cache2, "Linear 2%")
    assert layer1.get_display_cache("Linear 2%") is not None
    assert layer2.get_display_cache("Linear 2%") is not None

    # Test #1: Hover debounce
    window._on_pixel_hovered(10, 15)
    assert window._pending_hover_coords == (10, 15)
    assert window._hover_timer.isActive() is True
    # Trigger debounced handler directly
    window._do_pixel_hover()
    assert "10.00" in window.status_bar._lbl_pixel_values.text() or "20.00" in window.status_bar._lbl_pixel_values.text()

    # Test #2: Stretch changed invalidates display cache across ALL layers (layer2 is active so it gets freshly recomputed with Linear 5%)
    window._on_stretch_changed("Linear 5%")
    assert layer1.get_display_cache("Linear 2%") is None
    assert layer1.get_display_cache("Linear 5%") is None
    assert layer2.get_display_cache("Linear 2%") is None
    assert layer2.get_display_cache("Linear 5%") is not None

    # Test #3: Layer removal invalidates display cache
    layer1.set_display_cache(cache1, "Linear 5%")
    assert layer1._display_cache is not None
    window._on_layer_removed(layer1.layer_id)
    assert layer1._display_cache is None
    assert layer1.layer_id not in window._layers

    window.close()


def test_i18n_new_status_and_toolbox_keys(qapp):
    """Verify localized strings for status bar bands, tool selected, opened, and layer removed."""
    i18n.set_language("en")
    assert tr("status.bands_suffix") == "bands"
    assert tr("toolbox.msg_tool_selected").format(name="PCA") == "Tool selected: PCA"
    assert tr("status.msg_opened").format(name="test.tif") == "Opened: test.tif"
    assert tr("status.msg_layer_removed") == "Layer removed"

    status_bar = OpenENVIStatusBar()
    status_bar.update_pixel_values([1.0, 2.0, 3.0, 4.0, 5.0])
    assert "bands" in status_bar._lbl_pixel_values.text()

    i18n.set_language("zh")
    assert tr("status.bands_suffix") == "波段"
    assert tr("toolbox.msg_tool_selected").format(name="PCA") == "已选择工具: PCA"
    assert tr("status.msg_opened").format(name="test.tif") == "已打开: test.tif"
    assert tr("status.msg_layer_removed") == "图层已移除"

    status_bar.update_pixel_values([1.0, 2.0, 3.0, 4.0, 5.0])
    assert "波段" in status_bar._lbl_pixel_values.text()

    # Restore language to en
    i18n.set_language("en")


def test_classification_worker_and_progress(qapp):
    """Verify that ClassificationWorker runs off-thread, emits progress, and returns thematic layer."""
    from ui.dialogs.classification_dialog import ClassificationWorker
    from core.models import RasterMetadata
    from core.io.memory import MemoryRasterReader

    data = np.arange(100 * 100 * 3, dtype=np.float32).reshape(3, 100, 100)
    meta = RasterMetadata(
        width=100,
        height=100,
        bands=3,
        dtype="float32",
        crs="EPSG:4326",
    )
    reader = MemoryRasterReader(data, parent_metadata=meta)

    progress_updates = []
    results = []
    errors = []

    worker = ClassificationWorker(reader, total_bands=3, method_idx=0, num_classes=3, max_iters=5)
    worker.progress.connect(lambda pct, msg: progress_updates.append((pct, msg)))
    worker.finished.connect(lambda name, rgb: results.append((name, rgb)))
    worker.failed.connect(lambda err: errors.append(err))

    worker.run()  # run synchronously in test

    assert len(errors) == 0
    assert len(progress_updates) > 0
    # Final progress should reach 100%
    assert progress_updates[-1][0] == 100
    assert len(results) == 1
    layer_name, thematic_rgb = results[0]
    assert "K-Means" in layer_name
    assert thematic_rgb.shape == (100, 100, 3)


def test_memory_raster_reader_pixel_to_geo_robustness():
    """Verify that MemoryRasterReader.pixel_to_geo handles Affine, tuple, or invalid transforms safely."""
    from affine import Affine
    from core.models import RasterMetadata
    from core.io.memory import MemoryRasterReader

    data = np.zeros((1, 10, 10), dtype=np.float32)

    # 1. With proper Affine
    aff = Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)
    meta = RasterMetadata(width=10, height=10, bands=1, dtype="float32", transform=aff)
    reader = MemoryRasterReader(data, parent_metadata=meta)
    gx, gy = reader.pixel_to_geo(0, 0)
    assert gx is not None and gy is not None

    # 2. With tuple transform (which caused GCPTransformer ValueError in rasterio)
    tup = (30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)
    meta_tup = RasterMetadata(width=10, height=10, bands=1, dtype="float32", transform=tup)
    reader_tup = MemoryRasterReader(data, parent_metadata=meta_tup)
    gx2, gy2 = reader_tup.pixel_to_geo(0, 0)
    assert gx2 is not None and gy2 is not None
    assert gx2 == gx and gy2 == gy

    # 3. With invalid transform - should return (None, None) gracefully without crashing
    meta_bad = RasterMetadata(width=10, height=10, bands=1, dtype="float32", transform="invalid")
    reader_bad = MemoryRasterReader(data, parent_metadata=meta_bad)
    gx3, gy3 = reader_bad.pixel_to_geo(0, 0)
    assert gx3 is None and gy3 is None


def test_export_raster_dialog_ui(qapp, tmp_path):
    """Verify ExportRasterDialog initialization, band listing with wavelengths, and autoload."""
    from ui.dialogs.export_dialog import ExportRasterDialog
    from core.models import BandInfo, RasterLayer, RasterMetadata

    data = np.ones((2, 20, 20), dtype=np.float32)
    binfo = [
        BandInfo(index=0, name="Blue", wavelength=482.0, wavelength_unit="nm"),
        BandInfo(index=1, name="NIR", wavelength=865.0, wavelength_unit="nm"),
    ]
    meta = RasterMetadata(width=20, height=20, bands=2, dtype="float32", band_details=binfo)
    reader = MemoryRasterReader(data, parent_metadata=meta)
    layer = RasterLayer(layer_id="lyr_test", name="TestScene", file_path="mem://test", metadata=meta)

    dlg = ExportRasterDialog(layer, reader)
    assert dlg.cb_layers.count() == 1
    assert dlg.band_list.count() == 2
    # Check that wavelength is displayed in band item text
    assert "482.0 nm" in dlg.band_list.item(0).text()
    assert "865.0 nm" in dlg.band_list.item(1).text()
    # Check autoload checkbox is present and checked
    assert dlg.chk_autoload.isChecked()

    # Change format to ENVI
    dlg.cb_format.setCurrentIndex(1)
    assert dlg.cb_format.currentData() == "ENVI"
    assert not dlg.cb_compress.isEnabled()
    assert dlg.txt_path.text().endswith(".dat")

    dlg.close()


def test_resize_data_dialog_ui_and_roi_subset(qapp):
    """Verify ResizeDataDialog live dimensions preview, resampling dropdown, and ROI subsetting."""
    from ui.dialogs.resize_dialog import ResizeDataDialog
    from core.models import BandInfo, RasterLayer, RasterMetadata
    from core.roi import ROI

    data = np.ones((3, 100, 100), dtype=np.float32)
    binfo = [
        BandInfo(index=0, name="Red", wavelength=660.0),
        BandInfo(index=1, name="Green", wavelength=560.0),
        BandInfo(index=2, name="Blue", wavelength=480.0),
    ]
    meta = RasterMetadata(width=100, height=100, bands=3, dtype="float32", band_details=binfo)
    reader = MemoryRasterReader(data, parent_metadata=meta)

    # Attach an ROI to layer with bbox (10, 20, 50, 60) -> x0=10, y0=20, x1=50, y1=60
    roi = ROI(roi_id="roi_test", name="WaterBody", color="#0000ff", bbox=(10, 20, 50, 60))
    layer = RasterLayer(layer_id="lyr_resize", name="FullScene", file_path="mem://full", metadata=meta, rois=[roi])

    dlg = ResizeDataDialog(layer, reader)
    assert dlg.list_bands.count() == 3
    assert dlg.cb_method.count() == 3  # Nearest, Bilinear, Bicubic

    # Verify live preview text
    assert "100x100" in dlg.lbl_dims_info.text()
    assert "3 Bands" in dlg.lbl_dims_info.text()

    # Change scale factor to 0.5x
    dlg.cb_scale.setCurrentIndex(1)
    assert "50x50" in dlg.lbl_dims_info.text()

    # Test coordinate adjustments
    dlg.sp_xmin.setValue(10)
    dlg.sp_xmax.setValue(60)
    dlg.sp_ymin.setValue(20)
    dlg.sp_ymax.setValue(70)
    # Crop is 50x50, 0.5x scale -> 25x25
    assert "25x25" in dlg.lbl_dims_info.text()

    dlg.close()


def test_layer_stacking_dialog_ui_and_worker(qapp):
    """Verify LayerStackingDialog multi-selection, add all, and worker execution."""
    from ui.dialogs.stacking_dialog import LayerStackingDialog, StackingWorker
    from core.models import BandInfo, RasterLayer, RasterMetadata

    data = np.ones((2, 30, 30), dtype=np.float32)
    meta = RasterMetadata(width=30, height=30, bands=2, dtype="float32", band_details=[
        BandInfo(index=0, name="B1", wavelength=450.0),
        BandInfo(index=1, name="B2", wavelength=550.0),
    ])
    reader = MemoryRasterReader(data, parent_metadata=meta)
    layer = RasterLayer(layer_id="l1", name="Image1", file_path="mem://l1", metadata=meta)

    dlg = LayerStackingDialog({"l1": (layer, reader)})
    assert dlg.list_avail.count() == 2

    # Test Add All
    dlg._add_all_bands()
    assert dlg.list_selected.count() == 2
    assert "2" in dlg.lbl_selected_info.text()

    # Test worker directly
    worker = StackingWorker([(reader, 0), (reader, 1)], "StackedOutput", resampling_method="nearest")
    res = []
    worker.finished.connect(lambda name, cube, m: res.append((name, cube, m)))
    worker.run()

    assert len(res) == 1
    name, cube, m = res[0]
    assert name == "StackedOutput"
    assert cube.shape == (30, 30, 2)
    assert m.bands == 2
    assert m.band_details[0].wavelength == 450.0
    assert m.band_details[1].wavelength == 550.0

    dlg.close()

