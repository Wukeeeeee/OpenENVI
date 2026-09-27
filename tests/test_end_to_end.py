"""End-to-End Workflow Integration Tests for OpenENVI.

Simulates complete user workflows: opening datasets, RGB/Grayscale display,
pixel value/geo probing, interactive Z-Profile plotting, dynamic stretch,
Band Math, Spectral Indices, PCA, Classification, and ROI statistics.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.main_window import OpenENVIMainWindow
from core.events import event_bus
from core.roi import ROI
from core.synthetic import (
    generate_synthetic_cube,
    write_envi_dataset,
    write_geotiff_dataset,
)


@pytest.fixture(scope="module")
def sample_dataset():
    """Create a temporary synthetic ENVI dataset for workflow testing."""
    os.makedirs(r"E:\temp", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="openenvi_e2e_", dir=r"E:\temp")
    cube, wl, gt = generate_synthetic_cube(lines=32, samples=32, bands=16)
    base_path = os.path.join(tmp, "sample_hyperspectral")
    hdr_path, dat_path = write_envi_dataset(base_path, cube, wl, interleave="bsq")
    yield hdr_path, cube, wl
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_full_workflow(qapp, sample_dataset):
    """Execute end-to-end user workflows across all OpenENVI modules."""
    hdr_path, original_cube, original_wl = sample_dataset
    window = OpenENVIMainWindow()

    # 1. Open ENVI dataset
    layer = window.open_raster_file(hdr_path)
    assert layer is not None
    assert layer.name == "sample_hyperspectral.hdr"
    assert layer.metadata.bands == 16
    assert layer.metadata.width == 32
    assert layer.metadata.height == 32

    # Verify Docks state
    assert window.dock_layer_manager.tree.topLevelItemCount() >= 1
    assert window.dock_data_manager.tree.topLevelItemCount() >= 1

    # 2. Verify Grayscale and RGB display
    window.load_grayscale_band(layer.layer_id, 3)
    assert layer.display_mode == "grayscale"
    assert window.main_view._raw_data.shape == (32, 32)

    window.load_rgb_composition(layer.layer_id, 0, 1, 2)
    assert layer.display_mode == "rgb"
    assert window.main_view._raw_data.shape == (32, 32, 3)

    # 3. Dynamic Contrast Stretch
    window._on_stretch_changed("Linear 5%")
    assert window.main_view._stretch_mode == "Linear 5%"
    window._on_stretch_changed("Equalization")
    assert window.main_view._stretch_mode == "Equalization"
    window._on_stretch_changed("Gaussian")
    assert window.main_view._stretch_mode == "Gaussian"

    # 4. Cursor Probing & Status Bar Feedback
    window._on_pixel_hovered(15, 20)
    assert "File:" in window.status_bar._lbl_file_coords.text()
    assert "Geo:" in window.status_bar._lbl_geo_coords.text()
    assert "Value:" in window.status_bar._lbl_pixel_values.text()

    # 5. Pixel Click -> Spectral Profile (Z-Profile)
    window._on_pixel_clicked(10, 10)
    assert "Position: (10, 10)" in window.dock_spectral_profile._lbl_info.text()
    assert "Bands: 16" in window.dock_spectral_profile._lbl_info.text()

    # 6. Derived Products: Band Math
    b0 = window._readers[layer.layer_id].read_band(0)
    b1 = window._readers[layer.layer_id].read_band(1)
    math_res = (b1 - b0) / (b1 + b0 + 1e-6)
    derived_layer = window.add_derived_layer("BandMath_Test", math_res)
    assert derived_layer is not None
    assert derived_layer.metadata.bands == 1

    # 7. Derived Products: ROI Mean Spectrum Overlay
    roi = ROI(roi_id="roi_test", name="Water Area", bbox=(0, 16, 16, 32))
    reader = window._readers[layer.layer_id]
    mean_spec = roi.calculate_mean_spectrum(reader)
    assert mean_spec is not None
    assert len(mean_spec) == 16

    window.dock_spectral_profile.add_spectrum_overlay(
        values=mean_spec,
        wavelengths=original_wl,
        name="Water Mean",
        color="#3498db",
    )

    # 7b. Interactive Polygon ROI Drawing
    poly_pts = [(4.0, 4.0), (12.0, 4.0), (8.0, 12.0)]
    window.main_view.start_drawing_polygon(color="#e74c3c")
    window.main_view.add_roi_overlay(poly_pts, color="#e74c3c")
    window.main_view.stop_drawing_polygon()
    poly_roi = ROI(roi_id="poly_test", name="Polygon Test", color="#e74c3c", polygon_points=poly_pts)
    poly_stats = poly_roi.calculate_statistics(b0)
    assert poly_stats["count"] > 0

    # 7c. Pan-Sharpening Image Fusion (32x32 MS + 64x64 Pan)
    from core.algorithms.pansharpen import brovey_pansharpen, gram_schmidt_pansharpen
    pan_hi = np.full((64, 64), 100.0, dtype=np.float32)
    ms_rgb = [reader.read_band(0), reader.read_band(1), reader.read_band(2)]
    fused_rgb = brovey_pansharpen(pan_hi, ms_rgb)
    assert fused_rgb.shape == (64, 64, 3)
    fused_layer = window.add_derived_layer("PanSharpen_Fused", fused_rgb)
    assert fused_layer.metadata.width == 64
    assert fused_layer.metadata.height == 64

    # 7d. Radiometric Calibration
    from core.algorithms.radiometry import execute_calibration
    cal_res = execute_calibration(reader, cal_type="reflectance", custom_mult=2e-5, custom_add=-0.1, custom_sun_elev=45.0)
    cal_layer = window.add_derived_layer("Calibrated_TOA", cal_res)
    assert cal_layer.metadata.bands == 16

    # 7e. Export Raster to Disk
    from core.io.writer import export_raster
    export_out = os.path.join(r"E:\temp", "e2e_export_test.tif")
    if os.path.exists(export_out):
        os.remove(export_out)
    res_path = export_raster(reader, export_out, format="GTiff", band_indices=[0, 1, 2])
    assert os.path.exists(res_path)
    reopened_export = window.open_raster_file(res_path)
    assert reopened_export is not None
    assert reopened_export.metadata.bands == 3
    if os.path.exists(export_out):
        reopened_export = None
        window._on_layer_removed(f"layer_{len(window._layers)}_e2e_export_test.tif")

    # 8. Layer Removal and Canvas Clearing
    # Remove derived layers first
    for d_lyr in [fused_layer, cal_layer, derived_layer]:
        window._on_layer_removed(d_lyr.layer_id)

    # Only original layer remains
    assert len(window._layers) == 1
    assert window._active_layer_id == layer.layer_id
    assert window.main_view._raw_data is not None

    # Remove remaining layer -> Canvas, overview, and status bar must clear completely
    window._on_layer_removed(layer.layer_id)
    assert len(window._layers) == 0
    assert window._active_layer_id is None
    assert window.main_view._raw_data is None
    assert window.main_view.extent_roi.isVisible() is False
    assert "--, --" in window.status_bar._lbl_file_coords.text()

    # 9. Reset Layout & Close
    window._reset_dock_layout()
    window.close()


def test_e2e_bilingual_switch_live(qapp, sample_dataset):
    """Verify live bilingual language switching with active window, layers, and status bar."""
    from core.i18n import i18n, tr

    hdr_path, _, _ = sample_dataset
    window = OpenENVIMainWindow()
    layer = window.open_raster_file(hdr_path)
    assert layer is not None

    # Switch to Chinese
    i18n.set_language("zh")
    assert i18n.current_language == "zh"
    assert tr("app.ready") == "就绪"
    assert "OpenENVI" in window.windowTitle()
    assert window.dock_layer_manager.windowTitle() == tr("dock.layer_manager")
    assert window.dock_data_manager.windowTitle() == tr("dock.data_manager")
    assert window.dock_toolbox.windowTitle() == tr("dock.toolbox")
    assert window.dock_spectral_profile.windowTitle() == tr("dock.spectral_profile")
    assert tr("status.prefix_file") in window.status_bar._lbl_file_coords.text()
    assert tr("status.prefix_geo") in window.status_bar._lbl_geo_coords.text()
    assert tr("status.pixel_value") in window.status_bar._lbl_pixel_values.text()

    # Switch back to English
    i18n.set_language("en")
    assert i18n.current_language == "en"
    assert tr("app.ready") == "Ready"
    assert window.dock_layer_manager.windowTitle() == tr("dock.layer_manager")
    assert window.dock_data_manager.windowTitle() == tr("dock.data_manager")
    assert window.dock_toolbox.windowTitle() == tr("dock.toolbox")
    assert window.dock_spectral_profile.windowTitle() == tr("dock.spectral_profile")
    assert tr("status.prefix_file") in window.status_bar._lbl_file_coords.text()
    assert tr("status.prefix_geo") in window.status_bar._lbl_geo_coords.text()
    assert tr("status.pixel_value") in window.status_bar._lbl_pixel_values.text()

    window.close()


def test_e2e_roi_full_lifecycle(qapp, sample_dataset):
    """Verify complete ROI lifecycle: multi-polygon creation, stats, spectra, mask, and json round-trip."""
    import json
    from ui.dialogs.roi_dialog import ROIToolDialog

    hdr_path, _, original_wl = sample_dataset
    window = OpenENVIMainWindow()
    layer = window.open_raster_file(hdr_path)
    reader = window._readers[layer.layer_id]

    roi_dlg = ROIToolDialog(layer, reader, main_view=window.main_view, parent=window)

    # 1. Create a class and add multiple polygons
    roi_dlg._add_roi_class()
    roi = roi_dlg.rois[-1]
    poly1 = [(2.0, 2.0), (8.0, 2.0), (5.0, 8.0)]
    poly2 = [(10.0, 10.0), (16.0, 10.0), (13.0, 16.0)]
    roi.add_polygon(poly1)
    roi.add_polygon(poly2)
    assert roi.num_polygons == 2

    # 2. Multi-band statistics
    roi_dlg.table.selectRow(len(roi_dlg.rois) - 1)
    roi_dlg._compute_multiband_stats()
    assert hasattr(roi_dlg, "_stats_dlg")
    assert roi_dlg._stats_dlg.table.rowCount() == layer.metadata.bands

    # 3. Mean spectrum projection
    received_spectra = []
    roi_dlg.plot_mean_spectrum_requested.connect(
        lambda spec, wl, name, col: received_spectra.append((spec, wl, name, col))
    )
    roi_dlg._plot_mean_spectrum()
    assert len(received_spectra) == 1
    assert len(received_spectra[0][0]) == layer.metadata.bands

    # 4. Generate binary mask layer
    created_masks = []
    roi_dlg.mask_generated.connect(
        lambda name, mask, meta: created_masks.append((name, mask))
    )
    roi_dlg._create_mask_layer()
    assert len(created_masks) == 1
    assert created_masks[0][1].shape == (32, 32)
    assert np.max(created_masks[0][1]) == 255

    # 5. Export and import JSON
    tmp_json = os.path.join(tempfile.gettempdir(), "test_e2e_roi_export.json")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in roi_dlg.rois], f)
    
    with open(tmp_json, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    loaded_rois = [ROI.from_dict(d) for d in loaded_data]
    assert len(loaded_rois) == len(roi_dlg.rois)
    assert loaded_rois[-1].num_polygons == 2
    if os.path.exists(tmp_json):
        os.remove(tmp_json)

    # 6. Undo polygon & delete class
    roi.remove_last_polygon()
    assert roi.num_polygons == 1
    roi_dlg._delete_roi()

    roi_dlg.close()
    window.close()


def test_e2e_toolbox_pipeline(qapp, sample_dataset):
    """Verify algorithm tool pipeline integration and layer rendering."""
    from core.algorithms.classification import kmeans_clustering
    from core.algorithms.color import hsv_to_rgb, rgb_to_hsv
    from core.algorithms.continuum import continuum_removal_1d
    from core.algorithms.indices import calculate_ndvi, evaluate_band_math
    from core.algorithms.spectral import compute_pca

    hdr_path, _, original_wl = sample_dataset
    window = OpenENVIMainWindow()
    layer = window.open_raster_file(hdr_path)
    reader = window._readers[layer.layer_id]

    # NDVI
    b0 = reader.read_band(0)
    b1 = reader.read_band(1)
    ndvi = calculate_ndvi(b1, b0)
    l_ndvi = window.add_derived_layer("E2E_NDVI", ndvi)
    assert l_ndvi.name == "E2E_NDVI"

    # Band Math
    b_math = evaluate_band_math("b1 * 2.0 - b0", {"b0": b0, "b1": b1})
    l_math = window.add_derived_layer("E2E_BandMath", b_math)
    assert l_math.name == "E2E_BandMath"

    # PCA
    cube = np.stack([reader.read_band(i) for i in range(reader.metadata.bands)], axis=0)
    scores, evals, exp_var = compute_pca(cube, num_components=3)
    l_pca = window.add_derived_layer("E2E_PCA1", scores[0])
    assert l_pca.name == "E2E_PCA1"

    # K-Means
    c_map, centers = kmeans_clustering(cube, num_classes=3, max_iter=5)
    l_kmeans = window.add_derived_layer("E2E_KMeans", c_map.astype(np.float32))
    assert l_kmeans.name == "E2E_KMeans"

    # Continuum removal
    spec = cube[:, 16, 16]
    cr_spec, hull = continuum_removal_1d(spec, original_wl)
    assert len(cr_spec) == len(spec)
    assert np.all(cr_spec <= 1.0001)

    # Color space
    b2 = reader.read_band(2)
    hsv, hsv_names = rgb_to_hsv(b0, b1, b2, normalize_inputs=True)
    assert hsv.shape == (32, 32, 3)
    rgb_back, rgb_names = hsv_to_rgb(hsv[..., 0], hsv[..., 1], hsv[..., 2])
    assert rgb_back.shape == (32, 32, 3)

    window.close()


def test_e2e_synthetic_generator_dialog(qapp):
    """Verify synthetic dataset generator dialog and direct loading."""
    from ui.dialogs.synthetic_dialog import SyntheticDataDialog

    window = OpenENVIMainWindow()
    dlg = SyntheticDataDialog(parent=window)
    dlg.spin_size.setValue(32)
    dlg.spin_bands.setValue(8)

    generated_paths = []
    dlg.dataset_generated.connect(lambda p: generated_paths.append(p))

    # Trigger generation
    dlg._generate()
    assert len(generated_paths) == 1
    gen_path = generated_paths[0]
    assert os.path.exists(gen_path)

    # Load into window
    layer = window.open_raster_file(gen_path)
    assert layer is not None
    assert layer.metadata.bands == 8
    assert layer.metadata.width == 32
    assert layer.metadata.height == 32

    # Clean up generated file and header
    window._on_layer_removed(layer.layer_id)
    window.close()
    if os.path.exists(gen_path):
        os.remove(gen_path)
    base_no_ext = os.path.splitext(gen_path)[0]
    for ext in [".hdr", ".dat", ".raw"]:
        f_p = base_no_ext + ext
        if os.path.exists(f_p):
            os.remove(f_p)

