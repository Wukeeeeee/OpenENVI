"""Unit tests for raster export engine (GeoTIFF and ENVI Standard)."""

import os
import pytest
import numpy as np

from core.io.memory import MemoryRasterReader
from core.io.writer import export_raster
from core.io.reader import open_raster
from core.models import BandInfo, RasterMetadata


def test_export_raster_geotiff(tmp_path):
    """Test exporting MemoryRasterReader to GeoTIFF and re-opening."""
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    reader = MemoryRasterReader(data, name="Test Layer")

    out_file = str(tmp_path / "exported.tif")
    progress_calls = []

    res = export_raster(
        reader=reader,
        output_path=out_file,
        format="GTiff",
        progress_callback=lambda cur, tot: progress_calls.append((cur, tot)),
    )

    assert os.path.exists(res)
    assert len(progress_calls) == 1
    assert progress_calls[0] == (1, 1)

    # Re-open with open_raster and check values
    reopened = open_raster(res)
    assert reopened.metadata.width == 10
    assert reopened.metadata.height == 10
    assert reopened.metadata.bands == 1

    read_back = reopened.read_band(0)
    np.testing.assert_allclose(read_back, data, rtol=1e-5)
    reopened.close()


def test_export_raster_envi(tmp_path):
    """Test exporting multi-band MemoryRasterReader to ENVI Standard and re-opening."""
    b1 = np.ones((12, 8), dtype=np.float32) * 5.0
    b2 = np.ones((12, 8), dtype=np.float32) * 10.0
    cube = np.stack([b1, b2], axis=0)  # (2, 12, 8)

    reader = MemoryRasterReader(cube, name="Cube Layer")
    out_file = str(tmp_path / "exported_envi.dat")

    res = export_raster(
        reader=reader,
        output_path=out_file,
        format="ENVI",
    )

    assert os.path.exists(res)
    # Check that .hdr exists
    hdr_path = str(tmp_path / "exported_envi.hdr")
    assert os.path.exists(hdr_path)

    # Re-open with open_raster
    reopened = open_raster(out_file)
    assert reopened.metadata.width == 8
    assert reopened.metadata.height == 12
    assert reopened.metadata.bands == 2

    np.testing.assert_allclose(reopened.read_band(0), b1, rtol=1e-5)
    np.testing.assert_allclose(reopened.read_band(1), b2, rtol=1e-5)
    reopened.close()
