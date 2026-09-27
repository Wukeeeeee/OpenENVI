"""OpenENVI Raster Reader Factory & Dispatcher.

Inspects file extensions, companion headers, and metadata to dispatch
the optimal reader (ENVI or GeoTIFF) for the specified path.
"""

import os
from typing import Optional
from core.io.base import BaseRasterReader
from core.io.envi import ENVIRasterReader
from core.io.geotiff import GeoTIFFRasterReader
from core.io.landsat import LandsatMTLReader


def open_raster(file_path: str) -> BaseRasterReader:
    """Open and return a raster reader for the specified file.

    Args:
        file_path: Path to the image, header, or satellite metadata file.

    Returns:
        Instance of BaseRasterReader (ENVIRasterReader, GeoTIFFRasterReader, or LandsatMTLReader).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raster file not found: {file_path}")

    # Check if a directory was passed (e.g. uncompressed Landsat package folder)
    if os.path.isdir(file_path):
        for fname in os.listdir(file_path):
            if fname.lower().endswith(("_mtl.txt", "mtl.txt")):
                return LandsatMTLReader(os.path.join(file_path, fname))
        raise ValueError(f"Directory '{file_path}' does not contain recognized satellite metadata (*_MTL.txt).")

    base, ext = os.path.splitext(file_path)
    ext_lower = ext.lower()

    # Landsat Product Metadata (*_MTL.txt)
    if file_path.lower().endswith(("_mtl.txt", "mtl.txt")):
        return LandsatMTLReader(file_path)

    if ext_lower == ".txt":
        # Peek at text header to identify Landsat metadata
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                header_head = f.read(512)
            if "GROUP = L1_METADATA_FILE" in header_head or "LANDSAT_PRODUCT_ID" in header_head:
                return LandsatMTLReader(file_path)
        except Exception:
            pass

    # Direct ENVI header
    if ext_lower == ".hdr":
        return ENVIRasterReader(file_path)

    # Check companion .hdr for ENVI binary files (.dat, .raw, .img, .bsq, .bil, .bip)
    companion_candidates = [
        base + ".hdr",
        base + ".HDR",
        file_path + ".hdr",
        file_path + ".HDR",
    ]
    for cand in companion_candidates:
        if os.path.isfile(cand):
            return ENVIRasterReader(file_path)

    # Standard GeoTIFF / ERDAS IMAGINE / GDAL formats (.tif, .tiff, .img, etc.)
    gdal_error: Optional[Exception] = None
    try:
        return GeoTIFFRasterReader(file_path)
    except Exception as e:
        gdal_error = e

    # Fallback to ENVI reader attempt
    try:
        return ENVIRasterReader(file_path)
    except Exception as envi_error:
        if ext_lower in (".img", ".dat", ".raw", ".bin"):
            raise ValueError(
                f"Failed to open '{os.path.basename(file_path)}':\n"
                f"- If this is an ENVI binary image, please ensure its companion header file "
                f"('{os.path.basename(base)}.hdr') is in the same directory, or open the .hdr file directly.\n"
                f"- If this is an ERDAS IMAGINE (.img) file, GDAL reader reported: {gdal_error}"
            ) from envi_error
        raise ValueError(
            f"Unsupported or unreadable raster format for '{file_path}'. Error: {gdal_error or envi_error}"
        ) from envi_error
