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
    # If user provided .hdr as target path for ENVI, redirect binary data to .dat
    base, ext = os.path.splitext(output_path)
    if format.upper() in ("ENVI", "HDR") and ext.lower() == ".hdr":
        output_path = base + ".dat"

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
    # A big-endian ENVI source advertises its byte order in the dtype string
    # ('>i2'), which rasterio rejects outright as "invalid dtype". Drop the byte
    # order so the export writes native-order data.
    out_dtype = np.dtype(dtype or meta.dtype or "float32").newbyteorder("=").name

    # Normalize CRS
    crs_obj = None
    if meta.crs:
        try:
            crs_obj = CRS.from_string(meta.crs)
        except Exception:
            crs_obj = None

    # Normalize Transform
    transform_obj = None
    if isinstance(meta.transform, Affine):
        transform_obj = meta.transform
    elif meta.transform and hasattr(meta.transform, "__len__") and len(meta.transform) >= 6:
        try:
            transform_obj = Affine(*meta.transform[:6])
        except Exception:
            transform_obj = None

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

    effective_nodata = nodata if nodata is not None else meta.nodata
    if effective_nodata is not None:
        creation_options["nodata"] = effective_nodata

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

            # Cast data type if requested with safe NaN/Inf replacement
            if str(band_array.dtype) != out_dtype:
                if np.issubdtype(band_array.dtype, np.floating) and out_dtype in ("uint8", "uint16", "int16"):
                    fill_val = effective_nodata if effective_nodata is not None else 0
                    nan_mask = ~np.isfinite(band_array)
                    if np.any(nan_mask):
                        band_array = np.where(nan_mask, fill_val, band_array)

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

            # Set band description and wavelength tags
            b_name = f"Band {b_idx + 1}"
            wl = None
            if meta.band_details and b_idx < len(meta.band_details):
                b_info = meta.band_details[b_idx]
                b_name = b_info.name or b_name
                wl = b_info.wavelength

            dst.set_band_description(out_idx + 1, b_name)
            if wl is not None:
                try:
                    dst.update_tags(out_idx + 1, WAVELENGTH=str(wl))
                except Exception:
                    pass

            if progress_callback:
                progress_callback(out_idx + 1, total_bands)

    # If ENVI format, enrich .hdr with wavelengths, fwhm, and default bands
    if driver == "ENVI":
        hdr_path = os.path.splitext(output_path)[0] + ".hdr"
        if os.path.exists(hdr_path):
            _enrich_envi_header(hdr_path, meta, band_indices, effective_nodata)

    return output_path


def _enrich_envi_header(
    hdr_path: str,
    meta,
    band_indices: List[int],
    nodata: Optional[float] = None,
) -> None:
    """Enrich auto-generated ENVI header with wavelength and metadata fields."""
    try:
        with open(hdr_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        additions = []

        # Wavelengths & units
        if "wavelength =" not in content and meta.band_details:
            wls = []
            fwhms = []
            wl_unit = "nm"
            for b in band_indices:
                if b < len(meta.band_details) and meta.band_details[b].wavelength is not None:
                    wls.append(meta.band_details[b].wavelength)
                    wl_unit = meta.band_details[b].wavelength_unit or wl_unit
                    if meta.band_details[b].fwhm is not None:
                        fwhms.append(meta.band_details[b].fwhm)

            if len(wls) == len(band_indices):
                unit_str = "Nanometers" if "nm" in wl_unit.lower() else ("Micrometers" if "um" in wl_unit.lower() or "µm" in wl_unit.lower() else wl_unit)
                additions.append(f"wavelength units = {unit_str}")
                wl_lines = ",\n ".join(f"{w:.4f}" for w in wls)
                additions.append(f"wavelength = {{\n {wl_lines}\n}}")
                if len(fwhms) == len(band_indices):
                    fwhm_lines = ",\n ".join(f"{f:.4f}" for f in fwhms)
                    additions.append(f"fwhm = {{\n {fwhm_lines}\n}}")

        # Default bands (1-based for ENVI)
        if "default bands =" not in content and meta.default_bands:
            # Map original default bands into 1-based indices in output
            mapped = []
            for ob in meta.default_bands:
                if ob in band_indices:
                    mapped.append(str(band_indices.index(ob) + 1))
            if len(mapped) == 3:
                additions.append(f"default bands = {{ {', '.join(mapped)} }}")

        # Band names
        if "band names =" not in content and meta.band_details:
            b_names = []
            for b in band_indices:
                if b < len(meta.band_details) and meta.band_details[b].name:
                    b_names.append(meta.band_details[b].name)
                else:
                    b_names.append(f"Band {b + 1}")
            if b_names:
                names_str = ",\n ".join(b_names)
                additions.append(f"band names = {{\n {names_str}\n}}")

        # Classification header fields
        raw = getattr(meta, "raw_header", {}) or {}
        for key in ("file type", "classes", "class lookup", "class names"):
            if f"{key} =" not in content and key in raw:
                val = raw[key]
                if isinstance(val, (list, tuple)):
                    val_str = ",\n ".join(str(v) for v in val)
                    additions.append(f"{key} = {{\n {val_str}\n}}")
                else:
                    additions.append(f"{key} = {val}")

        # Data ignore value (NoData)
        if "data ignore value" not in content and nodata is not None:
            additions.append(f"data ignore value = {nodata}")

        if additions:
            with open(hdr_path, "a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(additions) + "\n")
    except Exception:
        pass
