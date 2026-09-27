"""Automated Tests for Landsat Product Package Reader.

Verifies MTL parsing, multi-band packaging, wavelength assignment,
lazy windowed pixel probing, and directory-based dataset opening.
"""

import os
import pytest
import numpy as np

from core.io.landsat import LandsatMTLReader, parse_mtl_file
from core.io.reader import open_raster

REAL_LANDSAT_PATH = r"E:\NDM下载\Compressed\LC81220442021019LGN00\LC08_L1TP_122044_20210119_20210119_01_RT_MTL.txt"


def test_parse_mtl_file(tmp_path):
    """Verify parsing of hierarchical MTL text files."""
    mtl_content = """GROUP = L1_METADATA_FILE
  GROUP = METADATA_FILE_INFO
    ORIGIN = "Image courtesy of USGS"
    LANDSAT_PRODUCT_ID = "LC08_TEST"
  END_GROUP = METADATA_FILE_INFO
  GROUP = PRODUCT_METADATA
    SPACECRAFT_ID = "LANDSAT_8"
    FILE_NAME_BAND_1 = "test_b1.tif"
  END_GROUP = PRODUCT_METADATA
END
"""
    mtl_file = tmp_path / "test_MTL.txt"
    mtl_file.write_text(mtl_content, encoding="utf-8")

    parsed = parse_mtl_file(str(mtl_file))
    assert "L1_METADATA_FILE" in parsed
    l1 = parsed["L1_METADATA_FILE"]
    assert l1["METADATA_FILE_INFO"]["LANDSAT_PRODUCT_ID"] == "LC08_TEST"
    assert l1["PRODUCT_METADATA"]["SPACECRAFT_ID"] == "LANDSAT_8"
    assert l1["PRODUCT_METADATA"]["FILE_NAME_BAND_1"] == "test_b1.tif"


@pytest.mark.skipif(not os.path.exists(REAL_LANDSAT_PATH), reason="Real Landsat 8 test data not present")
def test_real_landsat_8_mtl_reader():
    """Verify reading actual Landsat 8 scene package."""
    reader = open_raster(REAL_LANDSAT_PATH)
    assert isinstance(reader, LandsatMTLReader)

    meta = reader.metadata
    assert meta.width == 7531
    assert meta.height == 7691
    assert meta.bands == 10
    assert meta.default_bands == (3, 2, 1)

    # Verify band details
    assert meta.band_details[0].name == "Coastal Aerosol"
    assert meta.band_details[0].wavelength == 443.0
    assert meta.band_details[3].name == "Red"
    assert meta.band_details[3].wavelength == 655.0
    assert meta.band_details[4].name == "Near Infrared (NIR)"
    assert meta.band_details[4].wavelength == 865.0

    # Verify windowed pixel profile
    profile = reader.read_pixel_profile(500, 500)
    assert len(profile) == 10
    assert isinstance(profile, np.ndarray)

    # Verify geo conversion
    gx, gy = reader.pixel_to_geo(0, 0)
    assert gx is not None and gy is not None

    reader.close()


@pytest.mark.skipif(
    not os.path.exists(os.path.dirname(REAL_LANDSAT_PATH)),
    reason="Real Landsat 8 folder not present"
)
def test_landsat_directory_dispatch():
    """Verify opening Landsat dataset by directory path."""
    folder = os.path.dirname(REAL_LANDSAT_PATH)
    reader = open_raster(folder)
    assert isinstance(reader, LandsatMTLReader)
    assert reader.metadata.bands == 10
    reader.close()
