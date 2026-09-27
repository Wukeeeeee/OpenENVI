"""Unit tests for radiometric calibration and atmospheric correction."""

import pytest
import numpy as np

from core.algorithms.radiometry import (
    calibrate_band_to_reflectance,
    calibrate_band_to_radiance,
    execute_calibration,
)
from core.io.memory import MemoryRasterReader


def test_calibrate_band_to_reflectance():
    """Test DN to TOA reflectance conversion with sun angle."""
    # DN = 10000, mult = 2e-5, add = -0.1 -> rho_prime = 0.1
    # sun_elevation = 30 deg -> sin(30) = 0.5 -> rho_toa = 0.2
    raw_dn = np.full((10, 10), 10000.0, dtype=np.float32)
    refl = calibrate_band_to_reflectance(
        raw_dn,
        mult=2e-5,
        add=-0.1,
        sun_elevation_deg=30.0,
        apply_dos=False,
    )
    assert refl.shape == (10, 10)
    np.testing.assert_allclose(refl, 0.2, rtol=1e-4)


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
    # Band 1 -> 0.2
    np.testing.assert_allclose(res[..., 0], 0.2, rtol=1e-4)
    # Band 2: 15000 * 2e-5 - 0.1 = 0.2 -> / 0.5 = 0.4
    np.testing.assert_allclose(res[..., 1], 0.4, rtol=1e-4)
