"""OpenENVI Raster Export Engine.

Provides persistent exporting of in-memory and file-backed raster layers to
industry-standard GeoTIFF and ENVI Standard (.hdr + binary) formats.
Preserves geospatial coordinate reference systems, affine transforms, NoData values,
and spectral band nomenclature while streaming band-by-band to prevent memory spikes.
"""

import os
from typing import Callable, List, Optional
import numpy as np

import core.proj_setup  # Ensure PROJ library configuration is loaded
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from core.io.base import BaseRasterReader


def export_raster(
    reader: BaseRasterReader,
    output_path: str,
    format: str = "GTiff",
    band_indices: Optional[List[int]] = None,
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
    compress: Optional[str] = "lzw",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Export raster bands from a reader into a GeoTIFF or ENVI file.

    Args:
        reader: Source BaseRasterReader instance.
        output_path: Target file path on disk.
        format: Export driver, either 'GTiff' or 'ENVI'.
        band_indices: List of 0-based band indices to export. None for all bands.
        dtype: Output numpy/GDAL data type (e.g., 'float32', 'uint16', 'uint8').
               If None, uses reader metadata dtype or 'float32'.
        nodata: Nodata value to tag in the raster file.
        compress: Compression algorithm for GTiff ('lzw', 'deflate', or None).
        progress_callback: Optional callback receiving (completed_band_count, total_bands).

    Returns:
        The normalized absolute output path of the created file.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    meta = reader.metadata
    width = meta.width
    height = meta.height

    if band_indices is None:
        band_indices = list(range(meta.bands))

    if not band_indices:
        raise ValueError("Cannot export raster with 0 bands specified.")

    total_bands = len(band_indices)
    out_dtype = dtype or meta.dtype or "float32"

    # Normalize CRS
    crs_obj = None
    if meta.crs:
        try:
            crs_obj = CRS.from_string(meta.crs)
        except Exception:
            crs_obj = None

    # Normalize Transform
    transform_obj = None
    if meta.transform and len(meta.transform) >= 6:
        transform_obj = Affine(*meta.transform[:6])

    # Determine driver
    driver = "ENVI" if format.upper() in ("ENVI", "HDR") else "GTiff"

    # Prepare driver-specific creation options
    creation_options = {
        "driver": driver,
        "width": width,
        "height": height,
        "count": total_bands,
        "dtype": out_dtype,
        "crs": crs_obj,
        "transform": transform_obj,
    }

    if nodata is not None:
        creation_options["nodata"] = nodata
    elif meta.nodata is not None:
        creation_options["nodata"] = meta.nodata

    if driver == "GTiff":
        if compress and compress.lower() not in ("none", "no"):
            creation_options["compress"] = compress.lower()
        if width >= 256 and height >= 256:
            creation_options["tiled"] = True
            creation_options["blockxsize"] = 256
            creation_options["blockysize"] = 256

    # Write bands sequentially
    with rasterio.open(output_path, "w", **creation_options) as dst:
        for out_idx, b_idx in enumerate(band_indices):
            band_array = reader.read_band(b_idx)

            # Cast data type if requested
            if str(band_array.dtype) != out_dtype:
                if out_dtype == "uint8":
                    band_array = np.clip(band_array, 0, 255).astype(np.uint8)
                elif out_dtype == "uint16":
                    band_array = np.clip(band_array, 0, 65535).astype(np.uint16)
                elif out_dtype == "int16":
                    band_array = np.clip(band_array, -32768, 32767).astype(np.int16)
                else:
                    band_array = band_array.astype(out_dtype)

            # Rasterio 1-based band indexing
            dst.write(band_array, out_idx + 1)

            # Set band description if available
            if meta.band_details and b_idx < len(meta.band_details):
                b_name = meta.band_details[b_idx].name
                if b_name:
                    dst.set_band_description(out_idx + 1, b_name)

            if progress_callback:
                progress_callback(out_idx + 1, total_bands)

    return output_path
