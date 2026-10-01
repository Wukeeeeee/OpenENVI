"""Unit tests for radiometric calibration and atmospheric correction."""

import pytest
import numpy as np

from core.algorithms.radiometry import (
    calibrate_band_to_reflectance,
    calibrate_band_to_radiance,
    execute_calibration,
)
from core.io.memory import MemoryRasterReader
from core.models import BandInfo, RasterMetadata


def test_calibrate_band_to_reflectance():
    """Test DN to TOA reflectance conversion.

    The MTL coefficients already fold in 1/sin(sun_elevation), so the result is
    exactly mult * DN + add and the sun angle does not rescale it.
    """
    # DN = 10000, mult = 2e-5, add = -0.1 -> rho_toa = 0.1
    raw_dn = np.full((10, 10), 10000.0, dtype=np.float32)
    refl = calibrate_band_to_reflectance(
        raw_dn,
        mult=2e-5,
        add=-0.1,
        sun_elevation_deg=30.0,
        apply_dos=False,
    )
    assert refl.shape == (10, 10)
    np.testing.assert_allclose(refl, 0.1, rtol=1e-4)


def test_calibrate_band_to_reflectance_dos():
    """Test DOS-1 haze removal."""
    raw_dn = np.linspace(5000, 15000, 100, dtype=np.float32).reshape(10, 10)
    refl_dos = calibrate_band_to_reflectance(
        raw_dn,
        mult=2e-5,
        add=-0.1,
        sun_elevation_deg=45.0,
        apply_dos=True,
        dos_percentile=1.0,
    )
    assert refl_dos.shape == (10, 10)
    # The minimum value should be close to 0.0 after DOS
    assert refl_dos.min() >= 0.0
    assert refl_dos.min() < 0.05


def test_calibrate_band_to_radiance():
    """Test DN to radiance conversion."""
    raw_dn = np.full((5, 5), 500.0, dtype=np.float32)
    rad = calibrate_band_to_radiance(raw_dn, mult=0.5, add=10.0)
    np.testing.assert_allclose(rad, 260.0, rtol=1e-5)


def test_execute_calibration_memory_reader():
    """Test calibration execution across dataset bands."""
    b1 = np.full((10, 10), 10000.0, dtype=np.float32)
    b2 = np.full((10, 10), 15000.0, dtype=np.float32)
    cube = np.stack([b1, b2], axis=0)
    reader = MemoryRasterReader(cube, name="Test Layer")

    progress = []
    res = execute_calibration(
        reader=reader,
        cal_type="reflectance",
        custom_mult=2e-5,
        custom_add=-0.1,
        custom_sun_elev=30.0,
        progress_callback=lambda s, t: progress.append((s, t)),
    )

    assert res.shape == (10, 10, 2)
    assert len(progress) == 2
    # Band 1 -> 10000 * 2e-5 - 0.1 = 0.1
    np.testing.assert_allclose(res[..., 0], 0.1, rtol=1e-4)
    # Band 2 -> 15000 * 2e-5 - 0.1 = 0.2
    np.testing.assert_allclose(res[..., 1], 0.2, rtol=1e-4)


def test_landsat_band_resolution_prefers_specific_names_over_red():
    """Band names containing 'Infrared' must not be captured by the 'red' branch."""
    from core.algorithms.radiometry import extract_landsat_cal_params

    # Distinct coefficients per band so a mis-resolution is detectable
    expected_mult = {b: b / 100.0 for b in range(1, 12)}
    rescaling = {}
    for band in range(1, 12):
        rescaling[f"RADIANCE_MULT_BAND_{band}"] = band / 100.0
        rescaling[f"RADIANCE_ADD_BAND_{band}"] = -50.0 - band

    names = [
        "Coastal Aerosol", "Blue", "Green", "Red",
        "Near Infrared (NIR)", "Shortwave Infrared 1 (SWIR 1)",
        "Shortwave Infrared 2 (SWIR 2)", "Panchromatic", "Cirrus",
        "Thermal Infrared 1 (TIRS 1)", "Thermal Infrared 2 (TIRS 2)",
    ]
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    meta = RasterMetadata(
        width=4, height=4, bands=len(names), dtype=np.float32,
        band_details=[BandInfo(index=i, name=n) for i, n in enumerate(names)],
        raw_header={
            "mtl_data": {"L1_METADATA_FILE": {
                "RADIOMETRIC_RESCALING": rescaling,
                "IMAGE_ATTRIBUTES": {"SUN_ELEVATION": "30.0"},
            }},
            "sun_elevation": 30.0,
        },
    )
    reader = MemoryRasterReader(
        np.zeros((len(names), 4, 4), dtype=np.float32), name="Landsat", parent_metadata=meta
    )

    params = extract_landsat_cal_params(reader)
    assert params is not None
    for i, want in enumerate(expected):
        assert params["radiance_mult"][i] == pytest.approx(expected_mult[want]), names[i]
        assert params["radiance_add"][i] == pytest.approx(
            rescaling[f"RADIANCE_ADD_BAND_{want}"]
        ), names[i]


def test_execute_calibration_honors_custom_coefficients():
    """Explicit mult/add overrides must win even when MTL parameters are present."""
    meta = RasterMetadata(
        width=6, height=6, bands=1, dtype=np.float32,
        raw_header={
            "mtl_data": {"L1_METADATA_FILE": {
                "RADIOMETRIC_RESCALING": {
                    "RADIANCE_MULT_BAND_1": 0.01,
                    "RADIANCE_ADD_BAND_1": -50.0,
                },
                "IMAGE_ATTRIBUTES": {"SUN_ELEVATION": "30.0"},
            }},
            "sun_elevation": 30.0,
        },
    )
    cube = np.full((1, 6, 6), 1000.0, dtype=np.float32)
    reader = MemoryRasterReader(cube, name="Landsat", parent_metadata=meta)

    # Custom coefficients take precedence over the MTL-derived ones
    out = execute_calibration(
        reader, cal_type="radiance", custom_mult=1.0, custom_add=1000.0
    )
    np.testing.assert_allclose(out, 2000.0, rtol=1e-5)

    # Omitting them still uses the MTL values: 0.01 * 1000 - 50 = -40
    out_mtl = execute_calibration(reader, cal_type="radiance")
    np.testing.assert_allclose(out_mtl, 0.01 * 1000.0 - 50.0, rtol=1e-5)
