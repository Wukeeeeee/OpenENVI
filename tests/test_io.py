"""Tests for OpenENVI Raster I/O Engine and Synthetic Data Generator.

Verifies reading and writing of ENVI standard files (BSQ, BIL, BIP) and GeoTIFFs.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from core.io.reader import open_raster
from core.synthetic import (
    generate_synthetic_cube,
    write_envi_dataset,
    write_geotiff_dataset,
)


@pytest.fixture(scope="module")
def temp_io_dir():
    """Create a temporary directory for I/O test fixtures."""
    tmp = tempfile.mkdtemp(prefix="openenvi_io_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_envi_bsq_io(temp_io_dir):
    """Verify ENVI BSQ writing and reading."""
    cube, wl, gt = generate_synthetic_cube(lines=32, samples=32, bands=16)
    base_path = os.path.join(temp_io_dir, "test_bsq")
    hdr_path, dat_path = write_envi_dataset(base_path, cube, wl, interleave="bsq")

    assert os.path.isfile(hdr_path)
    assert os.path.isfile(dat_path)

    reader = open_raster(hdr_path)
    assert reader.metadata.width == 32
    assert reader.metadata.height == 32
    assert reader.metadata.bands == 16
    assert reader.metadata.interleave == "BSQ"
    assert len(reader.metadata.band_details) == 16

    # Test single band reading
    b0 = reader.read_band(0)
    assert b0.shape == (32, 32)
    np.testing.assert_allclose(b0, cube[0], rtol=1e-5)

    # Test pixel spectrum reading
    prof = reader.read_pixel_profile(10, 15)
    assert prof.shape == (16,)
    np.testing.assert_allclose(prof, cube[:, 15, 10], rtol=1e-5)

    # Test coordinate transformation (pixel centre, matching the GeoTIFF reader)
    geo_x, geo_y = reader.pixel_to_geo(0, 0)
    assert geo_x is not None and geo_y is not None
    assert geo_x == pytest.approx(500005.0)
    assert geo_y == pytest.approx(3499995.0)

    reader.close()


def test_envi_bil_io(temp_io_dir):
    """Verify ENVI BIL writing and reading."""
    cube, wl, gt = generate_synthetic_cube(lines=20, samples=20, bands=8)
    base_path = os.path.join(temp_io_dir, "test_bil")
    hdr_path, dat_path = write_envi_dataset(base_path, cube, wl, interleave="bil")

    reader = open_raster(dat_path)  # Open directly from .dat file
    assert reader.metadata.interleave == "BIL"

    b2 = reader.read_band(2)
    np.testing.assert_allclose(b2, cube[2], rtol=1e-5)

    prof = reader.read_pixel_profile(5, 5)
    np.testing.assert_allclose(prof, cube[:, 5, 5], rtol=1e-5)
    reader.close()


def test_envi_bip_io(temp_io_dir):
    """Verify ENVI BIP writing and reading."""
    cube, wl, gt = generate_synthetic_cube(lines=20, samples=20, bands=8)
    base_path = os.path.join(temp_io_dir, "test_bip")
    hdr_path, dat_path = write_envi_dataset(base_path, cube, wl, interleave="bip")

    reader = open_raster(hdr_path)
    assert reader.metadata.interleave == "BIP"

    b1 = reader.read_band(1)
    np.testing.assert_allclose(b1, cube[1], rtol=1e-5)

    prof = reader.read_pixel_profile(8, 8)
    np.testing.assert_allclose(prof, cube[:, 8, 8], rtol=1e-5)
    reader.close()


def test_geotiff_io(temp_io_dir):
    """Verify GeoTIFF writing and reading."""
    cube, wl, gt = generate_synthetic_cube(lines=30, samples=30, bands=4)
    tif_path = os.path.join(temp_io_dir, "test_geotiff.tif")
    write_geotiff_dataset(tif_path, cube, wl)

    assert os.path.isfile(tif_path)

    reader = open_raster(tif_path)
    assert reader.metadata.width == 30
    assert reader.metadata.height == 30
    assert reader.metadata.bands == 4

    b0 = reader.read_band(0)
    assert b0.shape == (30, 30)
    np.testing.assert_allclose(b0, cube[0], rtol=1e-5)

    prof = reader.read_pixel_profile(12, 14)
    assert prof.shape == (4,)
    np.testing.assert_allclose(prof, cube[:, 14, 12], rtol=1e-5)

    geo_x, geo_y = reader.pixel_to_geo(0, 0)
    assert geo_x is not None and geo_y is not None
    assert geo_x == pytest.approx(500005.0)  # pixel center
    assert geo_y == pytest.approx(3499995.0)

    reader.close()


def test_envi_img_io(temp_io_dir):
    """Verify ENVI binary dataset with .img extension and companion .hdr."""
    cube, wl, gt = generate_synthetic_cube(lines=20, samples=20, bands=5)
    base_path = os.path.join(temp_io_dir, "test_envi_img")
    hdr_path, dat_path = write_envi_dataset(base_path, cube, wl, interleave="bsq")

    # Rename .dat to .img
    img_path = base_path + ".img"
    if os.path.exists(img_path):
        os.remove(img_path)
    os.rename(dat_path, img_path)

    # Test opening both via .hdr and directly via .img
    reader_hdr = open_raster(hdr_path)
    assert reader_hdr.metadata.width == 20
    assert reader_hdr.metadata.height == 20
    assert reader_hdr.metadata.bands == 5
    reader_hdr.close()

    reader_img = open_raster(img_path)
    assert reader_img.metadata.width == 20
    assert reader_img.metadata.samples == 20
    assert reader_img.metadata.lines == 20
    assert reader_img.metadata.bands == 5
    b0 = reader_img.read_band(0)
    assert b0.shape == (20, 20)
    np.testing.assert_allclose(b0, cube[0], rtol=1e-5)
    reader_img.close()


def test_erdas_imagine_img_io(temp_io_dir):
    """Verify ERDAS IMAGINE (.img / HFA format) dataset reading via Rasterio."""
    import rasterio
    from rasterio.transform import from_origin

    cube = np.random.rand(3, 24, 24).astype(np.float32)
    erdas_path = os.path.join(temp_io_dir, "test_erdas.img")
    transform = from_origin(100.0, 50.0, 1.0, 1.0)

    with rasterio.open(
        erdas_path,
        "w",
        driver="HFA",
        height=24,
        width=24,
        count=3,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(cube)

    reader = open_raster(erdas_path)
    assert reader.metadata.width == 24
    assert reader.metadata.height == 24
    assert reader.metadata.bands == 3
    b1 = reader.read_band(1)
    assert b1.shape == (24, 24)
    np.testing.assert_allclose(b1, cube[1], rtol=1e-5)
    reader.close()
