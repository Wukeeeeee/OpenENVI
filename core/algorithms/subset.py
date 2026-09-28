"""OpenENVI Spatial and Spectral Subsetting & Resizing Engine.

Enables spatial cropping (bounding box), resolution downsampling/resizing,
and spectral band selection with geotransform synchronization.
"""

from typing import Callable, List, Optional, Tuple
import numpy as np

from rasterio.transform import Affine
from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterMetadata


def resize_subset_raster(
    reader: BaseRasterReader,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    selected_bands: List[int],
    scale_factor: float = 1.0,
    resample_method: str = "bilinear",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, RasterMetadata]:
    """Extract spatial subset with optional resolution scaling and spectral band selection.

    Args:
        reader: Source raster reader.
        x_min, x_max: Horizontal pixel bounds [x_min, x_max).
        y_min, y_max: Vertical line bounds [y_min, y_max).
        selected_bands: 0-indexed list of bands to extract.
        scale_factor: Spatial scale factor (e.g. 0.5 for 2x downsampling, 1.0 for original).
        resample_method: Interpolation mode ('nearest', 'bilinear', 'bicubic').
        progress_callback: Optional (current_band, total_bands) progress callback.

    Returns:
        Tuple of (subset_cube, subset_metadata)
    """
    orig_w = reader.metadata.width
    orig_h = reader.metadata.height

    # Clamp bounds to valid range
    x_min = max(0, min(orig_w - 1, x_min))
    x_max = max(x_min + 1, min(orig_w, x_max))
    y_min = max(0, min(orig_h - 1, y_min))
    y_max = max(y_min + 1, min(orig_h, y_max))

    crop_w = x_max - x_min
    crop_h = y_max - y_min

    target_w = max(1, int(round(crop_w * scale_factor)))
    target_h = max(1, int(round(crop_h * scale_factor)))

    total_bands = len(selected_bands)
    out_cube = np.empty((target_h, target_w, total_bands), dtype=np.float32)
    out_band_details: List[BandInfo] = []

    order_map = {"nearest": 0, "bilinear": 1, "bicubic": 3}
    resample_order = order_map.get(resample_method.lower(), 1)

    for idx, b_idx in enumerate(selected_bands):
        full_band = reader.read_band(b_idx)
        cropped = full_band[y_min:y_max, x_min:x_max]

        if (target_h, target_w) != (crop_h, crop_w):
            from core.algorithms.pansharpen import resample_band_to_grid
            resampled = resample_band_to_grid(cropped, target_h, target_w, order=resample_order)
            out_cube[:, :, idx] = resampled
        else:
            out_cube[:, :, idx] = cropped

        # Retain original band info
        b_name = f"Band {b_idx + 1}"
        wavelength = None
        wl_unit = "nm"
        fwhm = None
        if reader.metadata.band_details and b_idx < len(reader.metadata.band_details):
            binfo = reader.metadata.band_details[b_idx]
            b_name = binfo.name or b_name
            wavelength = binfo.wavelength
            wl_unit = binfo.wavelength_unit
            fwhm = binfo.fwhm

        out_band_details.append(
            BandInfo(index=idx, name=b_name, wavelength=wavelength, wavelength_unit=wl_unit, fwhm=fwhm)
        )

        if progress_callback:
            progress_callback(idx + 1, total_bands)

    # Compute updated geotransform if present
    new_transform = None
    if reader.metadata.transform is not None:
        try:
            if isinstance(reader.metadata.transform, Affine):
                aff = reader.metadata.transform
            else:
                aff = Affine(*reader.metadata.transform[:6])
            # Apply origin shift
            aff_sub = aff * Affine.translation(x_min, y_min)
            if scale_factor != 1.0:
                aff_sub = aff_sub * Affine.scale(1.0 / scale_factor, 1.0 / scale_factor)
            new_transform = aff_sub  # Retain as Affine instance
        except Exception:
            new_transform = reader.metadata.transform

    # Update raw ENVI header if present
    updated_header = dict(reader.metadata.raw_header) if reader.metadata.raw_header else {}
    if updated_header:
        updated_header["samples"] = target_w
        updated_header["lines"] = target_h
        updated_header["bands"] = total_bands
        if "band names" in updated_header and isinstance(updated_header["band names"], list):
            orig_names = updated_header["band names"]
            updated_header["band names"] = [orig_names[b] for b in selected_bands if b < len(orig_names)]
        if "wavelength" in updated_header and isinstance(updated_header["wavelength"], list):
            orig_wl = updated_header["wavelength"]
            updated_header["wavelength"] = [orig_wl[b] for b in selected_bands if b < len(orig_wl)]
        if "fwhm" in updated_header and isinstance(updated_header["fwhm"], list):
            orig_fwhm = updated_header["fwhm"]
            updated_header["fwhm"] = [orig_fwhm[b] for b in selected_bands if b < len(orig_fwhm)]

    # Map default bands
    new_default_bands = None
    if reader.metadata.default_bands:
        mapped_defaults = []
        for ob in reader.metadata.default_bands:
            if ob in selected_bands:
                mapped_defaults.append(selected_bands.index(ob))
        if len(mapped_defaults) == 3:
            new_default_bands = tuple(mapped_defaults)

    subset_meta = RasterMetadata(
        width=target_w,
        height=target_h,
        bands=total_bands,
        dtype="float32",
        crs=reader.metadata.crs,
        transform=new_transform,
        nodata=reader.metadata.nodata,
        band_details=out_band_details,
        raw_header=updated_header,
        default_bands=new_default_bands,
    )

    return out_cube, subset_meta
