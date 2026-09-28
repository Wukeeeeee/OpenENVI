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
    worker.finished.connect(lambda name, cmap, m, rgb: results.append((name, cmap, m, rgb)))
    worker.failed.connect(lambda err: errors.append(err))

    worker.run()  # run synchronously in test

    assert len(errors) == 0
    assert len(progress_updates) > 0
    # Final progress should reach 100%
    assert progress_updates[-1][0] == 100
    assert len(results) == 1
    layer_name, class_map, meta_out, thematic_rgb = results[0]
    assert "K-Means" in layer_name
    assert class_map.shape == (100, 100)
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


def test_band_math_advanced_syntax_and_dialog_ui(qapp):
    """Verify Band Math evaluation with float(), scalar division, comparison ops, and dialog."""
    from core.algorithms.indices import evaluate_band_math
    from ui.dialogs.band_math_dialog import BandMathDialog
    from core.models import BandInfo, RasterLayer, RasterMetadata

    b1 = np.full((10, 10), 10.0, dtype=np.float32)
    b2 = np.full((10, 10), 2.0, dtype=np.float32)
    vars_dict = {"b1": b1, "b2": b2}

    # 1. Scalar numerator division (previously crashed with ValueError non-broadcastable)
    res_div = evaluate_band_math("1.0 / b1", vars_dict)
    assert res_div.shape == (10, 10)
    np.testing.assert_allclose(res_div, 0.1, rtol=1e-5)

    # 2. ENVI IDL float() casting
    res_float = evaluate_band_math("(float(b1) - float(b2)) / (b1 + b2)", vars_dict)
    # (10 - 2) / (10 + 2) = 8 / 12 = 0.66667
    np.testing.assert_allclose(res_float, 8.0 / 12.0, rtol=1e-5)

    # 3. Comparison thresholding
    res_cmp = evaluate_band_math("(b1 > 5.0) * b2", vars_dict)
    np.testing.assert_allclose(res_cmp, 2.0, rtol=1e-5)

    # 4. Dialog UI
    meta = RasterMetadata(width=10, height=10, bands=2, dtype="float32", band_details=[
        BandInfo(index=0, name="Band 1"),
        BandInfo(index=1, name="Band 2"),
    ])
    reader = MemoryRasterReader(np.stack([b1, b2]), parent_metadata=meta)
    layer = RasterLayer(layer_id="l_math", name="MathLayer", file_path="mem://math", metadata=meta)

    dlg = BandMathDialog(layer, reader)
    assert dlg.combo_presets.count() > 3
    # Check variable mapping populated
    assert "b4" in dlg._var_combos or "b3" in dlg._var_combos
    # Insert custom expr
    dlg.edit_expr.setText("(b1 + b2) * 2.0")
    assert "b1" in dlg._var_combos
    assert "b2" in dlg._var_combos

    results = []
    dlg.result_generated.connect(lambda name, arr, m: results.append((name, arr, m)))
    dlg._execute()

    assert len(results) == 1
    name, arr, m = results[0]
    np.testing.assert_allclose(arr, 24.0, rtol=1e-5)
    assert m == meta
    dlg.close()


def test_indices_dialog_wavelength_micrometers_and_names(qapp):
    """Verify IndicesDialog automatic band selection with wavelengths in micrometers and names."""
    from ui.dialogs.indices_dialog import IndicesDialog
    from core.models import BandInfo, RasterLayer, RasterMetadata

    data = np.ones((5, 15, 15), dtype=np.float32)
    # Landsat-like bands with micrometers (0.48 um Blue, 0.56 um Green, 0.65 um Red, 0.86 um NIR, 2.2 um SWIR)
    binfo = [
        BandInfo(index=0, name="Coastal", wavelength=0.44, wavelength_unit="um"),
        BandInfo(index=1, name="Blue", wavelength=0.48, wavelength_unit="um"),
        BandInfo(index=2, name="Green", wavelength=0.56, wavelength_unit="um"),
        BandInfo(index=3, name="Red", wavelength=0.65, wavelength_unit="um"),
        BandInfo(index=4, name="NIR", wavelength=0.86, wavelength_unit="um"),
    ]
    meta = RasterMetadata(width=15, height=15, bands=5, dtype="float32", band_details=binfo)
    reader = MemoryRasterReader(data, parent_metadata=meta)
    layer = RasterLayer(layer_id="l_idx", name="LandsatScene", file_path="mem://idx", metadata=meta)

    dlg = IndicesDialog(layer, reader)
    # NDVI requires NIR (band index 4) and Red (band index 3)
    assert dlg.cb_nir.currentIndex() == 4
    assert dlg.cb_red.currentIndex() == 3

    # Switch to NDWI (Green: band 2, NIR: band 4)
    dlg.cb_index.setCurrentIndex(1)
    assert dlg.cb_green.currentIndex() == 2
    assert dlg.cb_nir.currentIndex() == 4

    # Compute NDVI
    results = []
    dlg.cb_index.setCurrentIndex(0)
    dlg.result_generated.connect(lambda name, arr, m: results.append((name, arr, m)))
    dlg._compute()

    assert len(results) == 1
    name, arr, m = results[0]
    assert "NDVI" in name
    assert arr.shape == (15, 15)
    assert m == meta
    dlg.close()


def test_pca_and_mnf_dialog_modes(qapp):
    """Verify PCADialog in both PCA and MNF modes with report generation and component outputs."""
    from ui.dialogs.pca_dialog import PCADialog
    from core.models import BandInfo, RasterLayer, RasterMetadata

    cube = np.random.RandomState(42).randn(6, 20, 20).astype(np.float32)
    meta = RasterMetadata(width=20, height=20, bands=6, dtype="float32")
    reader = MemoryRasterReader(cube, parent_metadata=meta)
    layer = RasterLayer(layer_id="l_tf", name="Hyperspectral", file_path="mem://hs", metadata=meta)

    # 1. PCA Mode
    dlg_pca = PCADialog(layer, reader, mode="pca")
    assert "PCA" in dlg_pca.windowTitle()
    res_pca = []
    dlg_pca.result_generated.connect(lambda n, a, m: res_pca.append((n, a, m)))
    dlg_pca._run_transform()
    assert len(res_pca) >= 1
    assert "PCA" in dlg_pca.txt_report.toPlainText()
    assert "Cumulative Variance" in dlg_pca.txt_report.toPlainText()
    dlg_pca.close()

    # 2. MNF Mode
    dlg_mnf = PCADialog(layer, reader, mode="mnf")
    assert "MNF" in dlg_mnf.windowTitle()
    res_mnf = []
    dlg_mnf.result_generated.connect(lambda n, a, m: res_mnf.append((n, a, m)))
    dlg_mnf._run_transform()
    assert len(res_mnf) >= 1
    assert "Minimum Noise Fraction" in dlg_mnf.txt_report.toPlainText()
    assert "Estimated SNR" in dlg_mnf.txt_report.toPlainText()
    dlg_mnf.close()


def test_sam_dialog_ui_and_worker(qapp):
    """Verify SAMDialog endmember selection from ROIs, thresholding, rule images, and worker."""
    from ui.dialogs.sam_dialog import SAMDialog, SAMWorker
    from core.models import BandInfo, RasterLayer, RasterMetadata
    from core.roi import ROI

    # 8-band cube with distinct vegetation and water spectral signatures
    cube = np.zeros((8, 20, 20), dtype=np.float32)
    # Band 3: Red (low veg, med water)
    # Band 4: NIR (high veg, low water)
    cube[3, :10, :10] = 0.05  # Veg Red
    cube[4, :10, :10] = 0.60  # Veg NIR
    cube[3, 10:, 10:] = 0.15  # Water Red
    cube[4, 10:, 10:] = 0.02  # Water NIR

    meta = RasterMetadata(width=20, height=20, bands=8, dtype="float32", band_details=[
        BandInfo(index=i, name=f"Band {i+1}", wavelength=400.0 + i * 100.0) for i in range(8)
    ])
    reader = MemoryRasterReader(cube, parent_metadata=meta)

    # Create 2 ROIs: Veg (top-left) and Water (bottom-right)
    roi_veg = ROI(roi_id="r_veg", name="Vegetation", color="#00ff00", bbox=(0, 0, 5, 5))
    roi_water = ROI(roi_id="r_water", name="Water", color="#0000ff", bbox=(12, 12, 18, 18))

    layer = RasterLayer(layer_id="l_sam", name="Scene", file_path="mem://sam", metadata=meta, rois=[roi_veg, roi_water])

    # Test Worker directly
    worker = SAMWorker(
        reader=reader,
        parent_meta=meta,
        selected_rois=[roi_veg, roi_water],
        max_angle=0.20,
        generate_rules=True,
        base_name="Scene",
    )
    res = []
    worker.finished.connect(
        lambda name, cmap, m, thematic, rules, r_meta: res.append((name, cmap, m, thematic, rules, r_meta))
    )
    worker.run()

    assert len(res) == 1
    class_name, class_map, m, thematic_rgb, rules, r_meta = res[0]
    assert "SAM Classify" in class_name
    assert class_map.shape == (20, 20)
    assert thematic_rgb.shape == (20, 20, 3)
    # Top-left pixel should match vegetation (green: #00ff00 -> [0, 255, 0])
    np.testing.assert_array_equal(thematic_rgb[2, 2], [0, 255, 0])
    # Bottom-right pixel should match water (blue: #0000ff -> [0, 0, 255])
    np.testing.assert_array_equal(thematic_rgb[15, 15], [0, 0, 255])
    # Unclassified regions or background should be black (0, 0, 0)
    np.testing.assert_array_equal(thematic_rgb[0, 19], [0, 0, 0])

    # Rule images check
    assert rules is not None
    assert rules.shape == (2, 20, 20)
    assert r_meta.bands == 2

    # Test Dialog UI
    dlg = SAMDialog(layer, reader)
    assert dlg.table_rois.rowCount() == 2
    assert dlg.btn_run.isEnabled()
    assert dlg.chk_rule_images.isChecked()
    dlg.close()


def test_pansharpen_worker_metadata_preservation(qapp):
    """Verify PanSharpenWorker preserves MS band details and PAN spatial transform."""
    from ui.dialogs.pansharpen_dialog import PanSharpenWorker
    from core.models import BandInfo, RasterMetadata
    from core.io.memory import MemoryRasterReader

    # 4-band MS image (10x10)
    ms_bands = [
        BandInfo(index=0, name="Blue", wavelength=480.0, wavelength_unit="nm", fwhm=30.0),
        BandInfo(index=1, name="Green", wavelength=560.0, wavelength_unit="nm", fwhm=40.0),
        BandInfo(index=2, name="Red", wavelength=660.0, wavelength_unit="nm", fwhm=40.0),
        BandInfo(index=3, name="NIR", wavelength=860.0, wavelength_unit="nm", fwhm=50.0),
    ]
    ms_data = np.ones((4, 10, 10), dtype=np.float32)
    ms_meta = RasterMetadata(
        width=10, height=10, bands=4, band_details=ms_bands, crs="EPSG:32650", transform=(30.0, 0, 1000, 0, -30.0, 2000)
    )
    ms_reader = MemoryRasterReader(ms_data, parent_metadata=ms_meta)

    # 1-band PAN image (20x20)
    pan_data = np.ones((1, 20, 20), dtype=np.float32) * 1.5
    pan_meta = RasterMetadata(
        width=20, height=20, bands=1, crs="EPSG:32650", transform=(15.0, 0, 1000, 0, -15.0, 2000)
    )
    pan_reader = MemoryRasterReader(pan_data, parent_metadata=pan_meta)

    worker = PanSharpenWorker(
        ms_reader=ms_reader,
        selected_bands=[1, 2, 3],  # Green, Red, NIR
        pan_reader=pan_reader,
        pan_file_path=None,
        method="gs",
        name="PanSharpen_Test",
    )

    results = []
    worker.finished.connect(lambda name, fused, meta: results.append((name, fused, meta)))
    worker.run()

    assert len(results) == 1
    out_name, out_fused, out_meta = results[0]
    assert out_name == "PanSharpen_Test"
    assert out_fused.shape == (20, 20, 3)
    assert out_meta.width == 20
    assert out_meta.height == 20
    assert out_meta.bands == 3
    assert out_meta.transform == (15.0, 0, 1000, 0, -15.0, 2000)
    assert len(out_meta.band_details) == 3
    assert out_meta.band_details[0].name == "Green"
    assert out_meta.band_details[0].wavelength == 560.0
    assert out_meta.band_details[1].name == "Red"
    assert out_meta.band_details[1].wavelength == 660.0
    assert out_meta.band_details[2].name == "NIR"
    assert out_meta.band_details[2].wavelength == 860.0


def test_radiometry_worker_subset_metadata_preservation(qapp):
    """Verify RadiometryWorker creates cal_meta containing only the selected calibrated bands."""
    from ui.dialogs.radiometry_dialog import RadiometryWorker
    from core.models import BandInfo, RasterMetadata
    from core.io.memory import MemoryRasterReader

    b_info = [
        BandInfo(index=0, name="Coastal", wavelength=443.0),
        BandInfo(index=1, name="Blue", wavelength=482.0),
        BandInfo(index=2, name="Green", wavelength=562.0),
        BandInfo(index=3, name="Red", wavelength=655.0),
    ]
    data = np.full((4, 15, 15), 1000.0, dtype=np.float32)
    meta = RasterMetadata(width=15, height=15, bands=4, band_details=b_info, crs="EPSG:4326")
    reader = MemoryRasterReader(data, parent_metadata=meta)

    # Calibrate only Red (index 3) and Green (index 2)
    worker = RadiometryWorker(
        reader=reader,
        cal_type="TOA Reflectance",
        band_indices=[2, 3],
        custom_mult=0.0001,
        custom_add=0.0,
        custom_sun_elev=45.0,
        name="TOA_Refl_Sub",
    )

    results = []
    worker.finished.connect(lambda name, res, cal_meta: results.append((name, res, cal_meta)))
    worker.run()

    assert len(results) == 1
    name, res_arr, cal_meta = results[0]
    assert cal_meta.bands == 2
    assert len(cal_meta.band_details) == 2
    assert cal_meta.band_details[0].wavelength == 562.0
    assert "Green" in cal_meta.band_details[0].name
    assert cal_meta.band_details[1].wavelength == 655.0
    assert "Red" in cal_meta.band_details[1].name


def test_thematic_layer_display_cache_and_persistence():
    """Verify that thematic classification layer preserves display image across stretch mode changes."""
    from core.models import RasterLayer, RasterMetadata

    meta = RasterMetadata(width=10, height=10, bands=1)
    layer = RasterLayer(layer_id="l1", name="Thematic", file_path="mem://test", metadata=meta)

    dummy_rgb = np.full((10, 10, 3), 128, dtype=np.uint8)
    layer.set_thematic_image(dummy_rgb)

    assert layer.is_thematic is True
    # Cached display must be returned regardless of stretch mode
    assert layer.get_display_cache("linear2") is dummy_rgb
    assert layer.get_display_cache("equalize") is dummy_rgb
    assert layer.get_display_cache("gaussian") is dummy_rgb

    # When invalidate_display_cache is called across all layers on stretch change, thematic image must not be erased
    layer.invalidate_display_cache()
    assert layer.get_display_cache("min_max") is dummy_rgb


def test_raster_statistics_automatic_nodata():
    """Verify that calculate_raster_statistics respects reader.metadata.nodata automatically."""
    from core.algorithms.statistics import calculate_raster_statistics
    from core.models import RasterMetadata
    from core.io.memory import MemoryRasterReader

    # Array where half the pixels are nodata (-9999) and half are 10.0
    arr = np.array([
        [-9999.0, -9999.0, 10.0, 10.0],
        [-9999.0, -9999.0, 10.0, 10.0],
    ], dtype=np.float32)
    meta = RasterMetadata(width=4, height=2, bands=1, nodata=-9999.0)
    reader = MemoryRasterReader(arr, parent_metadata=meta)

    stats = calculate_raster_statistics(reader)
    assert len(stats) == 1
    b_stat = stats[0]
    assert b_stat["count"] == 4
    assert b_stat["min"] == 10.0
    assert b_stat["max"] == 10.0
    assert b_stat["mean"] == 10.0


def test_maxlik_dialog_workflow(qapp):
    """Verify MaximumLikelihoodDialog populates ROIs and executes background classification."""
    from ui.dialogs.maxlik_dialog import MaximumLikelihoodDialog
    from core.roi import ROI

    # Create 4-band test image (20x20)
    data = np.zeros((4, 20, 20), dtype=np.float32)
    data[:, :, :10] = 50.0
    data[:, :, 10:] = 200.0

    meta = RasterMetadata(width=20, height=20, bands=4)
    reader = MemoryRasterReader(data, parent_metadata=meta)
    layer = RasterLayer(layer_id="test_mlc", name="Test MLC Layer", file_path="memory://test_mlc", metadata=meta)

    # Add 2 ROIs: Class 0 (left) and Class 1 (right)
    roi1 = ROI(roi_id="roi_0", name="Water", color="#0000ff", bbox=(0, 0, 5, 5))
    roi2 = ROI(roi_id="roi_1", name="Land", color="#00ff00", bbox=(12, 12, 18, 18))
    layer.rois = [roi1, roi2]

    dlg = MaximumLikelihoodDialog(layer=layer, reader=reader)
    assert dlg.table_rois.rowCount() == 2
    assert dlg.btn_run.isEnabled() is True

    # Test ROI selection controls
    dlg._deselect_all_rois()
    assert len(dlg._get_selected_rois()) == 0
    dlg._select_all_rois()
    assert len(dlg._get_selected_rois()) == 2

    # Run worker directly
    results = []
    dlg.result_generated.connect(lambda name, cmap, meta, rgb: results.append((name, cmap, rgb)))

    # Execute classification worker
    dlg._run_classification()
    assert dlg._worker is not None
    dlg._worker.wait(10000)

    # Process events to deliver finished signal
    from PySide6.QtCore import QCoreApplication
    QCoreApplication.processEvents()

    assert len(results) >= 1
    layer_name, cmap, rgb = results[0]
    assert "MLC Classify" in layer_name
    assert cmap.shape == (20, 20)
    assert rgb.shape == (20, 20, 3)
    assert rgb.dtype == np.uint8


def test_data_manager_close_file_signal(qapp):
    """Verify DataManager close_file_requested signal removes layer and releases resources."""
    from app.main_window import OpenENVIMainWindow
    from PySide6.QtWidgets import QTreeWidgetItem

    win = OpenENVIMainWindow()
    meta = RasterMetadata(width=10, height=10, bands=3)
    reader = MemoryRasterReader(np.ones((3, 10, 10), dtype=np.float32), parent_metadata=meta)
    layer = RasterLayer(layer_id="close_test_id", name="Close Test", file_path="memory://close_test", metadata=meta)

    win._readers["close_test_id"] = reader
    win._layers["close_test_id"] = layer
    win._active_layer_id = "close_test_id"
    win.dock_layer_manager.add_layer(layer)
    win.dock_data_manager.add_dataset(layer)

    assert "close_test_id" in win._layers
    assert "close_test_id" in win._readers

    # Trigger close file signal
    win.dock_data_manager.close_file_requested.emit("close_test_id")

    assert "close_test_id" not in win._layers
    assert "close_test_id" not in win._readers
    assert win._active_layer_id is None




