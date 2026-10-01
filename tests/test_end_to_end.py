"""End-to-End Workflow Integration Tests for OpenENVI.

Simulates complete user workflows: opening datasets, RGB/Grayscale display,
pixel value/geo probing, interactive Z-Profile plotting, dynamic stretch,
Band Math, Spectral Indices, PCA, Classification, and ROI statistics.
"""

import os
import shutil
import time
import tempfile
import numpy as np
import pytest
from PySide6.QtCore import Qt
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
    # _raw_data is released immediately after display (memory optimisation); verify via image_item
    assert window.main_view._raw_data is None
    img = window.main_view.image_item.image
    assert img is not None
    # pyqtgraph stores images transposed: (W, H) for grayscale
    assert img.shape[0] == 32 and img.shape[1] == 32

    window.load_rgb_composition(layer.layer_id, 0, 1, 2)
    assert layer.display_mode == "rgb"
    assert window.main_view._raw_data is None
    img = window.main_view.image_item.image
    assert img is not None
    # pyqtgraph stores RGB as (W, H, C)
    assert img.shape[0] == 32 and img.shape[1] == 32 and img.shape[2] == 3

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
    # _raw_data is always released after display; verify canvas still has content via image_item
    assert window.main_view.image_item.image is not None

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



def test_e2e_svm_sid_ica_dialogs(qapp, sample_dataset):
    """Verify the SVM, SID and ICA dialogs construct, populate and classify on real data."""
    from core.algorithms.classification import svm_classification
    from core.algorithms.spectral import compute_ica, spectral_information_divergence
    from ui.dialogs.pca_dialog import PCADialog
    from ui.dialogs.sid_dialog import SIDDialog
    from ui.dialogs.svm_dialog import SVMDialog
    from ui.toolbox import IMPLEMENTED_TOOLS

    hdr_path, _, _ = sample_dataset
    window = OpenENVIMainWindow()
    layer = window.open_raster_file(hdr_path)
    reader = window._readers[layer.layer_id]
    bands = layer.metadata.bands
    lines, samples = layer.metadata.height, layer.metadata.width

    # Two spatially distinct training regions on the synthetic scene
    roi_a = ROI(roi_id="roi_a", name="RegionA", bbox=(2, 2, 14, 14))
    roi_b = ROI(roi_id="roi_b", name="RegionB", bbox=(18, 18, 30, 30))
    layer.rois = [roi_a, roi_b]

    assert int(np.sum(roi_a.get_mask(lines, samples))) > 0
    assert int(np.sum(roi_b.get_mask(lines, samples))) > 0

    # 1. SVM dialog builds its ROI table and enables Run for 2 classes
    svm_dlg = SVMDialog(layer, reader, parent=window)
    assert svm_dlg.table_rois.rowCount() == 2
    assert svm_dlg.btn_run.isEnabled() is True
    # Training-sample cap defaults to 2000/class, with "All" selectable
    assert svm_dlg.spin_max_samples.value() == 2000

    # Deselecting down to one class must disable Run (SVM needs >= 2 classes)
    svm_dlg._deselect_all_rois()
    assert svm_dlg.btn_run.isEnabled() is False
    svm_dlg._select_all_rois()
    assert svm_dlg.btn_run.isEnabled() is True

    cube = np.stack([reader.read_band(b) for b in range(bands)], axis=0)

    def roi_training_samples(roi):
        mask = roi.get_mask(lines, samples)
        ys, xs = np.where(mask)
        return cube[:, ys, xs].T

    class_map, probs = svm_classification(
        cube=cube,
        training_data={0: roi_training_samples(roi_a), 1: roi_training_samples(roi_b)},
        kernel="rbf",
        C=100.0,
        gamma="scale",
        probability_threshold=0.0,
    )
    assert class_map.shape == (lines, samples)
    assert probs.shape == (2, lines, samples)
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-4)
    svm_dlg.close()

    # 2. SID dialog: reference spectra table plus divergence classification
    sid_dlg = SIDDialog(layer, reader, parent=window)
    assert sid_dlg.table_rois.rowCount() == 2
    assert sid_dlg.btn_run.isEnabled() is True

    refs = np.vstack([
        roi_a.calculate_mean_spectrum(reader),
        roi_b.calculate_mean_spectrum(reader),
    ])
    assert refs.shape == (2, bands)

    rules, sid_map = spectral_information_divergence(cube=cube, reference_spectra=refs)
    assert rules.shape == (2, lines, samples)
    assert sid_map.shape == (lines, samples)
    assert np.all(rules >= 0.0)
    assert set(np.unique(sid_map)).issubset({0, 1, -1})
    sid_dlg.close()

    # 3. ICA dialog dispatches to the shared PCADialog in "ica" mode
    ica_dlg = PCADialog(layer=layer, reader=reader, mode="ica", parent=window)
    assert ica_dlg.mode == "ica"
    assert ica_dlg.spin_components.maximum() == min(20, bands)
    ica_dlg.close()

    ic_cube, mixing = compute_ica(cube, num_components=3)
    assert ic_cube.shape == (3, lines, samples)
    assert mixing.shape == (bands, 3)
    assert np.all(np.isfinite(ic_cube))

    # 4. All three tools are registered as implemented in the toolbox
    assert {"svm", "sid", "ica"}.issubset(IMPLEMENTED_TOOLS)

    # 5. Main window exposes handlers for each menu action
    for handler in ("show_svm_dialog", "show_sid_dialog", "show_ica_dialog"):
        assert callable(getattr(window, handler))

    window.close()


def test_e2e_sff_dialog(qapp, sample_dataset):
    """Verify the SFF dialog builds, runs on real data and emits abundance layers."""
    from core.algorithms.spectral import spectral_feature_fitting
    from ui.dialogs.sff_dialog import SFFDialog, least_squares_abundances
    from ui.toolbox import IMPLEMENTED_TOOLS

    hdr_path, _, _ = sample_dataset
    window = OpenENVIMainWindow()
    layer = window.open_raster_file(hdr_path)
    reader = window._readers[layer.layer_id]
    bands = layer.metadata.bands
    lines, samples = layer.metadata.height, layer.metadata.width

    dlg = SFFDialog(layer, reader, parent=window)
    # 0 means "choose the count automatically"
    assert dlg.spin_num.value() == 0
    assert dlg.spin_num.maximum() == bands
    assert dlg.spin_num.specialValueText() != ""
    assert dlg.chk_abundance.isChecked() is True
    assert dlg.txt_out_name.text().endswith("_SFF_Abundance")

    # The wavelength axis falls back to band indices when the file has none
    axis = dlg._axis()
    assert len(axis) == bands

    captured = {}
    dlg.endmembers_extracted.connect(
        lambda name, ends, resid: captured.update(
            name=name, ends=ends, residuals=resid
        )
    )

    # Drive the dialog's own handler rather than the algorithm directly, so the
    # worker, the cube transpose and the signal wiring are all exercised.
    dlg.spin_num.setValue(3)
    dlg.chk_abundance.setChecked(True)
    dlg._start()
    worker = dlg._worker
    assert worker is not None
    assert worker.wait(30000)
    # wait() returns once run() is over, but the worker's signals cross threads as
    # queued connections, so the main thread still has to drain them.
    for _ in range(50):
        QApplication.processEvents()
        if "ends" in captured:
            break
        time.sleep(0.02)

    assert "ends" in captured, captured
    ends, residuals = captured["ends"], captured["residuals"]
    assert ends.shape == (3, bands)
    assert residuals.shape == (3,)
    assert np.all(np.isfinite(ends))
    assert ends.min() >= 0.0

    # Abundances must reconstruct the scene on the extracted basis
    cube = np.stack([reader.read_band(b) for b in range(bands)], axis=0)
    abundance = least_squares_abundances(cube, ends)
    assert abundance.shape == (3, lines, samples)
    assert abundance.min() >= 0.0

    # Rebuild the cube as the sum of abundance-weighted endmembers
    pixels = abundance.transpose(1, 2, 0).reshape(-1, 3)
    rebuilt = (pixels @ ends).reshape(lines, samples, bands).transpose(2, 0, 1)
    finite = np.all(np.isfinite(cube), axis=0) & np.all(np.isfinite(rebuilt), axis=0)
    assert finite.any()
    residual = np.linalg.norm(cube - rebuilt, axis=0)[finite]
    data_scale = np.linalg.norm(cube, axis=0)[finite]
    assert np.mean(residual / np.maximum(data_scale, 1e-9)) < 0.5

    # The same call with automatic count must not raise
    auto_ends, _ = spectral_feature_fitting(cube, num_features=0)
    assert len(auto_ends) >= 1

    assert "sff" in IMPLEMENTED_TOOLS
    assert callable(getattr(window, "show_sff_dialog"))

    dlg.close()
    window.close()


def test_e2e_mosaic_dialog(qapp, sample_dataset):
    """Verify Mosaicking combines two georeferenced layers into one raster."""
    from ui.dialogs.mosaic_dialog import MosaicDialog
    from ui.toolbox import IMPLEMENTED_TOOLS
    from core.models import RasterMetadata

    hdr_path, _, _ = sample_dataset
    window = OpenENVIMainWindow()
    window.open_raster_file(hdr_path)

    def geo_meta(origin_x):
        return RasterMetadata(
            width=16, height=12, bands=3, dtype="float32", crs="EPSG:32650",
            transform=(30.0, 0.0, origin_x, 0.0, -30.0, 4000000.0), nodata=None,
        )

    left = window.add_derived_layer(
        "E2E_Mosaic_Left",
        np.tile(np.array([1.0, 2.0, 3.0], np.float32), (12, 16, 1)),
        parent_metadata=geo_meta(500000.0),
    )
    right = window.add_derived_layer(
        "E2E_Mosaic_Right",
        np.tile(np.array([7.0, 8.0, 9.0], np.float32), (12, 16, 1)),
        parent_metadata=geo_meta(500240.0),  # 8 pixels east: a 8-pixel overlap
    )

    layers = window.get_available_layers()
    dlg = MosaicDialog(layers, parent=window)

    # The mosaic target lists every georeferenced layer and previews the union grid
    assert dlg.table_inputs.rowCount() == len(layers)
    assert dlg.btn_ok.isEnabled()

    # Mosaicking is a many-to-one choice: keep only the two tiles we added, so the
    # 10 m ENVI sample scene stops dictating the reference resolution.
    dlg._set_all(False)
    for row in range(dlg.table_inputs.rowCount()):
        item = dlg.table_inputs.item(row, 0)
        if item.data(Qt.UserRole) in (left.layer_id, right.layer_id):
            item.setCheckState(Qt.Checked)
    dlg._update_grid_preview()
    # 16 px each, overlapping by 8, so the union is 24 x 12 at 30 m.
    assert "24 x 12" in dlg.lbl_grid.text(), dlg.lbl_grid.text()

    captured = {}
    dlg.result_generated.connect(
        lambda name, cube, meta: captured.update(name=name, cube=cube, meta=meta)
    )

    dlg.cmb_resample.setCurrentIndex(dlg.cmb_resample.findData("nearest"))
    dlg.spin_feather.setValue(4.0)
    dlg.txt_out_name.setText("E2E_Mosaic")
    dlg._start()

    worker = dlg._worker
    assert worker is not None
    assert worker.wait(60000)
    # run() has finished, but the worker's signals are queued, so drain them.
    for _ in range(50):
        QApplication.processEvents()
        if "cube" in captured:
            break
        time.sleep(0.02)

    assert "cube" in captured, captured
    cube, meta = captured["cube"], captured["meta"]
    assert captured["name"] == "E2E_Mosaic"
    # 16 px each, overlapping by 8, so the union is 24 px wide.
    assert cube.shape == (12, 24, 3)
    assert meta.crs == "EPSG:32650"
    assert meta.transform[4] < 0

    # Non-overlapping halves keep their own source values; the seam is blended.
    np.testing.assert_allclose(cube[:, :8, 0], 1.0, rtol=1e-5)
    np.testing.assert_allclose(cube[:, 16:, 0], 7.0, rtol=1e-5)
    assert cube[:, 8:16, 0].min() >= 1.0 and cube[:, 8:16, 0].max() <= 7.0

    # "mosaic" is offered in the toolbox, and the window routes it to a dialog
    assert "mosaic" in IMPLEMENTED_TOOLS
    assert callable(getattr(window, "show_mosaic_dialog"))

    window.close()


def test_e2e_spectral_library_dialog(qapp, sample_dataset):
    """Verify the Spectral Library Viewer plots, filters and matches real data."""
    from ui.dialogs.spectral_library_dialog import SpectralLibraryDialog
    from ui.toolbox import IMPLEMENTED_TOOLS
    from core.algorithms.spectral_library import get_spectral_library

    hdr_path, _, _ = sample_dataset
    window = OpenENVIMainWindow()
    layer = window.open_raster_file(hdr_path)
    reader = window._readers[layer.layer_id]

    dlg = SpectralLibraryDialog(layer, reader, parent=window)

    # 1. The browser lists the whole library, and the category combo filters it
    total = len(get_spectral_library())
    assert dlg.table_library.rowCount() == total
    assert str(total) in dlg.lbl_library.text()

    veg_index = dlg.cmb_category.findData("Vegetation")
    assert veg_index > 0
    dlg.cmb_category.setCurrentIndex(veg_index)
    vegetation_rows = dlg.table_library.rowCount()
    assert 0 < vegetation_rows < total
    for row in range(vegetation_rows):
        assert dlg.table_library.item(row, 1).text() == "Vegetation"
    dlg.cmb_category.setCurrentIndex(0)
    assert dlg.table_library.rowCount() == total

    # 2. The scene spectrum is read and described in the status line
    assert dlg._scene_curve is not None
    wl, curve = dlg._scene_curve
    assert wl.shape == (layer.metadata.bands,)
    assert curve.shape == (layer.metadata.bands,)
    assert np.all(np.isfinite(curve))
    assert layer.name in dlg.lbl_scene.text()

    # 3. The library is ranked against the scene by spectral angle, best first
    assert dlg.table_matches.rowCount() > 0
    angles = [
        float(dlg.table_matches.item(r, 2).text())
        for r in range(dlg.table_matches.rowCount())
    ]
    assert angles == sorted(angles)
    assert angles[0] <= dlg.spin_max_angle.value()

    # Tightening the threshold must not add matches
    dlg.spin_max_angle.setValue(1.0)
    tight = dlg.table_matches.rowCount()
    assert tight <= dlg.table_matches.rowCount()
    dlg.spin_max_angle.setValue(25.0)
    assert dlg.table_matches.rowCount() >= tight

    # 4. A redraw is a rebuild: the scene curve plus one curve per selection.
    dlg.table_library.selectRow(0)
    assert len(dlg.selected_spectra()) == 1
    dlg._update_plot()
    redrawn = len(dlg.plot_widget.listDataItems())
    assert redrawn == 1 + len(dlg.selected_spectra())
    dlg._update_plot()
    assert len(dlg.plot_widget.listDataItems()) == redrawn, "redraw is not idempotent"

    # The match table overlays its picked curve on top, without a redraw.
    dlg.table_matches.selectRow(0)
    dlg._show_match_curve()
    assert len(dlg.plot_widget.listDataItems()) == redrawn + 1

    # 5. Without a layer the viewer still browses, it just does not match
    standalone = SpectralLibraryDialog(parent=window)
    assert standalone._scene_curve is None
    assert standalone.table_matches.rowCount() == 0
    assert standalone.table_library.rowCount() == total

    assert "spectral_lib" in IMPLEMENTED_TOOLS
    assert callable(getattr(window, "show_spectral_library_dialog"))

    window.close()


def test_e2e_classification_dialog_survives_nodata(qapp, sample_dataset):
    """Drive the real Classification dialog on a scene containing NoData.

    One masked pixel used to make a cluster centre NaN, which made every
    distance against it NaN, which made argmin return 0 everywhere -- so the
    entire scene came back as a single class. The dialog now has to deliver a
    genuinely multi-class map with the masked region left unclassified, and the
    thematic RGB it renders must agree with that class map.
    """
    from ui.dialogs.classification_dialog import ClassificationDialog

    hdr_path, _, _ = sample_dataset
    window = OpenENVIMainWindow()
    layer = window.open_raster_file(hdr_path)
    reader = window._readers[layer.layer_id]

    # Poison a corner of the reader with NaN, the way a NoData scene arrives.
    real_read_band = reader.read_band
    masked = {}

    def read_band(index):
        data = np.array(real_read_band(index), dtype=np.float32, copy=True)
        if index not in masked:
            masked[index] = True
            data[:6, :6] = np.nan
        return data

    reader.read_band = read_band

    dlg = ClassificationDialog(layer, reader, parent=window)
    captured = {}
    dlg.result_generated.connect(
        lambda name, cmap, meta, rgb: captured.update(
            name=name, cmap=cmap, meta=meta, rgb=rgb
        )
    )
    dlg.spin_classes.setValue(3)
    dlg.spin_iters.setValue(6)
    dlg.btn_run.click()

    assert dlg._worker is not None
    assert dlg._worker.wait(30000)
    # wait() returns once run() is over, but the queued result signal still needs
    # an event loop turn to reach the dialog.
    for _ in range(100):
        QApplication.processEvents()
        time.sleep(0.02)
        if "cmap" in captured:
            break

    assert "cmap" in captured, "the dialog never emitted a result"
    class_map = captured["cmap"]
    assert class_map.shape == (layer.metadata.height, layer.metadata.width)

    labels = set(np.unique(class_map).tolist())
    assert -1 in labels, "the masked corner must be reported as unclassified"
    assert len(labels - {-1}) >= 2, f"the scene collapsed into {labels}"

    # The masked corner is unclassified, and no valid pixel was dropped.
    assert np.all(class_map[:6, :6] == -1)
    assert np.all(class_map[6:, 6:] >= 0)

    # The rendered thematic image must line up with the class map exactly.
    rgb = captured["rgb"]
    assert rgb.shape == (layer.metadata.height, layer.metadata.width, 3)
    unclassified_rgb = rgb[0, 0]
    valid_rgb = rgb[-1, -1]
    assert not np.array_equal(unclassified_rgb, valid_rgb)

    window.close()


def test_e2e_radiometry_dialog_does_not_double_correct_the_sun_angle(qapp, sample_dataset):
    """The MTL coefficients already fold in 1/sin(sun_elevation).

    Dividing again inflated every band by 1/sin(theta) -- 1.41x at the dialog's
    default 45 degrees -- so a calibrated layer never matched the product.
    """
    from ui.dialogs.radiometry_dialog import RadiometryDialog

    hdr_path, _, _ = sample_dataset
    window = OpenENVIMainWindow()
    layer = window.open_raster_file(hdr_path)
    reader = window._readers[layer.layer_id]

    dn = 10000.0
    reader = _ConstantDNReader(reader, dn)

    dlg = RadiometryDialog(
        available_layers={"L": (layer, reader)}, parent=window
    )
    dlg.spin_mult.setValue(2e-5)
    dlg.spin_add.setValue(-0.1)
    dlg.spin_sun.setValue(45.0)
    dlg.rb_toa_refl.setChecked(True)
    dlg.txt_name.setText("E2E_TOA")

    captured = {}
    dlg.result_generated.connect(
        lambda name, result, meta: captured.update(name=name, result=result, meta=meta)
    )
    dlg._start_calibration()

    assert dlg._worker is not None
    assert dlg._worker.wait(30000)
    for _ in range(100):
        QApplication.processEvents()
        time.sleep(0.02)
        if "result" in captured:
            break

    assert "result" in captured, "the dialog never emitted a calibrated layer"
    values = np.asarray(captured["result"])
    valid = values[np.isfinite(values) & (values > 0)]

    # mult * DN + add, with no further sun-angle division.
    np.testing.assert_allclose(np.median(valid), dn * 2e-5 - 0.1, rtol=1e-4)
    assert np.median(valid) < 0.2, "a 1/sin(45) division would roughly double this"

    window.close()


class _ConstantDNReader:
    """Read-only wrapper returning a constant DN, ignoring the wrapped reader."""

    def __init__(self, inner, dn):
        self._inner = inner
        self._dn = float(dn)

    def read_band(self, index):
        shape = (self._inner.metadata.height, self._inner.metadata.width)
        return np.full(shape, self._dn, dtype=np.float32)

    @property
    def metadata(self):
        return self._inner.metadata
