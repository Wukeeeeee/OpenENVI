"""Unit tests for spatial and spectral subsetting and resizing."""

import numpy as np
import pytest
from core.algorithms.subset import resize_subset_raster
from core.io.memory import MemoryRasterReader


def test_spatial_and_spectral_subset():
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    b1 = data
    b2 = data * 2.0
    b3 = data * 3.0
    cube = np.stack([b1, b2, b3], axis=0)
    reader = MemoryRasterReader(cube, name="TestScene")

    # Crop 2:8 (H) and 3:7 (W), select bands 0 and 2
    out_cube, meta = resize_subset_raster(
        reader=reader,
        x_min=3,
        x_max=7,
        y_min=2,
        y_max=8,
        selected_bands=[0, 2],
        scale_factor=1.0,
    )

    assert out_cube.shape == (6, 4, 2)
    assert meta.width == 4
    assert meta.height == 6
    assert meta.bands == 2
    # Band 0 slice check
    np.testing.assert_array_equal(out_cube[:, :, 0], b1[2:8, 3:7])
    # Band 2 slice check
    np.testing.assert_array_equal(out_cube[:, :, 1], b3[2:8, 3:7])


def test_resize_downsample_factor():
    b1 = np.ones((20, 20), dtype=np.float32) * 5.0
    reader = MemoryRasterReader(b1, name="ResampleScene")

    out_cube, meta = resize_subset_raster(
        reader=reader,
        x_min=0,
        x_max=20,
        y_min=0,
        y_max=20,
        selected_bands=[0],
        scale_factor=0.5,
    )

    assert out_cube.shape == (10, 10, 1)
    assert meta.width == 10
    assert meta.height == 10
    assert np.allclose(out_cube, 5.0)


def test_resize_subset_preserves_affine_and_resample_methods():
    """Verify that resize_subset_raster keeps transform as an Affine instance and supports nearest neighbor."""
    from affine import Affine
    from core.models import BandInfo, RasterMetadata

    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    aff = Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)
    binfo = [BandInfo(index=0, name="Red", wavelength=660.0)]
    orig_meta = RasterMetadata(
        width=10,
        height=10,
        bands=1,
        dtype="float32",
        transform=aff,
        crs="EPSG:32650",
        nodata=-9999.0,
        band_details=binfo,
        raw_header={"samples": 10, "lines": 10, "bands": 1, "wavelength": ["660.0"]},
    )
    reader = MemoryRasterReader(data, parent_metadata=orig_meta)

    # 1. Nearest neighbor test with crop and 2x downsample
    out_cube, meta = resize_subset_raster(
        reader=reader,
        x_min=2,
        x_max=8,
        y_min=2,
        y_max=8,
        selected_bands=[0],
        scale_factor=0.5,
        resample_method="nearest",
    )

    assert out_cube.shape == (3, 3, 1)
    assert meta.width == 3
    assert meta.height == 3
    assert meta.nodata == -9999.0
    # Must be Affine object (not tuple)
    assert isinstance(meta.transform, Affine)
    # Check origin shift: x_min=2, y_min=2 -> origin moves by 2 * 30m = 60m east, 2 * -30m = -60m north
    assert meta.transform.c == 500000.0 + 60.0
    assert meta.transform.f == 4000000.0 - 60.0
    # Check resolution scaled by 1/0.5 = 2.0x -> 60m pixels
    assert meta.transform.a == 60.0
    assert meta.transform.e == -60.0
    # Check raw header updated
    assert meta.raw_header["samples"] == 3
    assert meta.raw_header["lines"] == 3
    assert len(meta.band_details) == 1
    assert meta.band_details[0].wavelength == 660.0
