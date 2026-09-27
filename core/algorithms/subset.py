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
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, RasterMetadata]:
    """Extract spatial subset with optional resolution scaling and spectral band selection.

    Args:
        reader: Source raster reader.
        x_min, x_max: Horizontal pixel bounds [x_min, x_max).
        y_min, y_max: Vertical line bounds [y_min, y_max).
        selected_bands: 0-indexed list of bands to extract.
        scale_factor: Spatial scale factor (e.g. 0.5 for 2x downsampling, 1.0 for original).
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

    for idx, b_idx in enumerate(selected_bands):
        full_band = reader.read_band(b_idx)
        cropped = full_band[y_min:y_max, x_min:x_max]

        if (target_h, target_w) != (crop_h, crop_w):
            from core.algorithms.pansharpen import resample_band_to_grid
            resampled = resample_band_to_grid(cropped, target_h, target_w)
            out_cube[:, :, idx] = resampled
        else:
            out_cube[:, :, idx] = cropped

        # Retain original band info
        b_name = f"Band {b_idx + 1}"
        wavelength = None
        if reader.metadata.band_details and b_idx < len(reader.metadata.band_details):
            binfo = reader.metadata.band_details[b_idx]
            b_name = binfo.name or b_name
            wavelength = binfo.wavelength

        out_band_details.append(BandInfo(index=idx, name=b_name, wavelength=wavelength))

        if progress_callback:
            progress_callback(idx + 1, total_bands)

    # Compute updated geotransform if present
    new_transform = None
    if reader.metadata.transform is not None:
        try:
            aff = Affine(*reader.metadata.transform[:6])
            # Apply origin shift
            aff_sub = aff * Affine.translation(x_min, y_min)
            if scale_factor != 1.0:
                aff_sub = aff_sub * Affine.scale(1.0 / scale_factor, 1.0 / scale_factor)
            new_transform = tuple(aff_sub)
        except Exception:
            new_transform = reader.metadata.transform

    r_name = getattr(reader, "name", "") or getattr(reader, "_name", "") or "Subset"
    subset_meta = RasterMetadata(
        width=target_w,
        height=target_h,
        bands=total_bands,
        dtype="float32",
        crs=reader.metadata.crs,
        transform=new_transform,
        band_details=out_band_details,
        raw_header=reader.metadata.raw_header,
    )

    return out_cube, subset_meta
