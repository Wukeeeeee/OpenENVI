"""Unit tests for layer stacking engine."""

import numpy as np
import pytest
from core.algorithms.stacking import stack_bands
from core.io.memory import MemoryRasterReader


def test_stack_identical_dimensions():
    b1 = np.ones((20, 30), dtype=np.float32) * 1.0
    b2 = np.ones((20, 30), dtype=np.float32) * 2.0
    r1 = MemoryRasterReader(b1, name="Layer1")
    r2 = MemoryRasterReader(b2, name="Layer2")

    progress = []
    cube, meta = stack_bands(
        [(r1, 0), (r2, 0)],
        progress_callback=lambda c, t: progress.append((c, t)),
    )

    assert cube.shape == (20, 30, 2)
    assert meta.bands == 2
    assert np.all(cube[:, :, 0] == 1.0)
    assert np.all(cube[:, :, 1] == 2.0)
    assert progress == [(1, 2), (2, 2)]


def test_stack_different_dimensions_resamples():
    b1 = np.ones((20, 30), dtype=np.float32) * 5.0
    b2 = np.ones((10, 15), dtype=np.float32) * 10.0
    r1 = MemoryRasterReader(b1, name="RefLayer")
    r2 = MemoryRasterReader(b2, name="SmallLayer")

    cube, meta = stack_bands([(r1, 0), (r2, 0)])

    assert cube.shape == (20, 30, 2)
    assert meta.bands == 2
    assert np.allclose(cube[:, :, 0], 5.0)
    assert np.allclose(cube[:, :, 1], 10.0)


def test_stack_bands_preserves_wavelengths_and_reprojects():
    """Verify that stack_bands synthesizes band details, preserves wavelengths, and supports reprojection."""
    from affine import Affine
    from core.models import BandInfo, RasterMetadata

    aff_ref = Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)
    meta1 = RasterMetadata(
        width=10,
        height=10,
        bands=1,
        dtype="float32",
        transform=aff_ref,
        crs="EPSG:32650",
        band_details=[BandInfo(index=0, name="Coastal", wavelength=443.0, wavelength_unit="nm", fwhm=15.0)],
        raw_header={"samples": 10, "lines": 10, "bands": 1},
    )
    r1 = MemoryRasterReader(np.ones((10, 10), dtype=np.float32) * 2.0, name="L8", parent_metadata=meta1)

    aff_pan = Affine(15.0, 0.0, 500000.0, 0.0, -15.0, 4000000.0)
    meta2 = RasterMetadata(
        width=20,
        height=20,
        bands=1,
        dtype="float32",
        transform=aff_pan,
        crs="EPSG:32650",
        band_details=[BandInfo(index=0, name="Pan", wavelength=590.0, wavelength_unit="nm", fwhm=180.0)],
    )
    r2 = MemoryRasterReader(np.ones((20, 20), dtype=np.float32) * 8.0, name="Pan", parent_metadata=meta2)

    cube, meta = stack_bands([(r1, 0), (r2, 0)], resampling_method="bilinear")

    assert cube.shape == (10, 10, 2)
    assert meta.bands == 2
    assert isinstance(meta.transform, Affine)
    assert len(meta.band_details) == 2
    assert meta.band_details[0].wavelength == 443.0
    assert meta.band_details[1].wavelength == 590.0
    assert "L8" in meta.band_details[0].name
    assert "Pan" in meta.band_details[1].name
    assert np.allclose(cube[:, :, 0], 2.0)
    assert np.allclose(cube[:, :, 1], 8.0)
