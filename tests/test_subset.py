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
