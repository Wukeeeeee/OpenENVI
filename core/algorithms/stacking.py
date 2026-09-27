"""OpenENVI Layer Stacking Engine.

Combines multiple individual raster bands (from different datasets or files)
into a unified multi-band raster cube with spatial alignment and metadata synthesis.
"""

from typing import Callable, List, Optional, Tuple
import numpy as np

from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterMetadata


def stack_bands(
    band_sources: List[Tuple[BaseRasterReader, int]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, RasterMetadata]:
    """Stack specified bands from one or more readers into a single multi-band array.

    Args:
        band_sources: List of (reader, band_index) tuples in desired stack order.
        progress_callback: Optional (current, total) progress callback.

    Returns:
        Tuple of:
            - 3D numpy array of shape (height, width, len(band_sources))
            - Combined RasterMetadata with synthesized band details
    """
    if not band_sources:
        raise ValueError("No band sources provided for stacking.")

    # Reference grid is defined by the first band
    ref_reader, ref_band_idx = band_sources[0]
    ref_meta = ref_reader.metadata
    target_h = ref_meta.height
    target_w = ref_meta.width

    total_bands = len(band_sources)
    stacked_cube = np.empty((target_h, target_w, total_bands), dtype=np.float32)
    combined_band_details: List[BandInfo] = []

    for i, (reader, band_idx) in enumerate(band_sources):
        data = reader.read_band(band_idx)

        # Handle size discrepancy with nearest/bilinear resampling if needed
        if data.shape != (target_h, target_w):
            from core.algorithms.pansharpen import resample_band_to_grid
            data = resample_band_to_grid(data, target_h, target_w)

        stacked_cube[:, :, i] = data

        # Extract band name and wavelength
        r_name = getattr(reader, "name", "") or getattr(reader, "_name", "") or "Layer"
        band_name = f"Band {i + 1}"
        wavelength = None
        if reader.metadata.band_details and band_idx < len(reader.metadata.band_details):
            binfo = reader.metadata.band_details[band_idx]
            band_name = f"{r_name}: {binfo.name}" if binfo.name else f"{r_name}: B{band_idx + 1}"
            wavelength = binfo.wavelength
        else:
            band_name = f"{r_name}: B{band_idx + 1}"

        combined_band_details.append(BandInfo(index=i, name=band_name, wavelength=wavelength))

        if progress_callback:
            progress_callback(i + 1, total_bands)

    combined_metadata = RasterMetadata(
        width=target_w,
        height=target_h,
        bands=total_bands,
        dtype="float32",
        crs=ref_meta.crs,
        transform=ref_meta.transform,
        band_details=combined_band_details,
        raw_header=ref_meta.raw_header,
    )

    return stacked_cube, combined_metadata
