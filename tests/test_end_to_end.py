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
