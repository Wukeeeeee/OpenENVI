"""OpenENVI Raster Statistics Engine.

Computes comprehensive statistical metrics per band:
- Min, Max, Mean, Standard Deviation
- Histogram distributions
- Memory-efficient streaming calculation across all bands
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np

from core.io.base import BaseRasterReader


def calculate_band_statistics(
    band_data: np.ndarray,
    nodata: Optional[float] = None,
    bins: int = 100,
) -> Dict[str, Any]:
    """Calculate descriptive statistics and histogram for a single band slice.

    Args:
        band_data: 2D numpy array.
        nodata: Optional nodata value to ignore.
        bins: Number of histogram bins.

    Returns:
        Dict containing:
            - 'count': Valid pixel count
            - 'min': Minimum valid value
            - 'max': Maximum valid value
            - 'mean': Mean valid value
            - 'std': Standard deviation
            - 'hist_counts': Array of counts per bin
            - 'bin_edges': Array of bin edges (len = bins + 1)
    """
    valid_mask = np.isfinite(band_data)
    if nodata is not None:
        valid_mask &= (band_data != nodata)

    valid_pixels = band_data[valid_mask]
    if len(valid_pixels) == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "hist_counts": np.zeros(bins, dtype=np.int64),
            "bin_edges": np.linspace(0.0, 1.0, bins + 1),
        }

    min_val = float(np.min(valid_pixels))
    max_val = float(np.max(valid_pixels))
    mean_val = float(np.mean(valid_pixels))
    std_val = float(np.std(valid_pixels))

    if min_val == max_val:
        hist_counts = np.zeros(bins, dtype=np.int64)
        hist_counts[0] = len(valid_pixels)
        bin_edges = np.linspace(min_val - 0.5, max_val + 0.5, bins + 1)
    else:
        hist_counts, bin_edges = np.histogram(valid_pixels, bins=bins, range=(min_val, max_val))

    return {
        "count": int(len(valid_pixels)),
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "hist_counts": hist_counts,
        "bin_edges": bin_edges,
    }


def calculate_raster_statistics(
    reader: BaseRasterReader,
    nodata: Optional[float] = None,
    bins: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Stream-calculate descriptive statistics for all bands in a raster dataset.

    Args:
        reader: BaseRasterReader instance.
        nodata: Optional nodata value.
        bins: Number of histogram bins.
        progress_callback: Optional (current_band, total_bands) callback.

    Returns:
        List of dicts containing statistics for each band (0 to total_bands - 1).
    """
    total_bands = reader.metadata.bands
    results = []
    effective_nodata = nodata if nodata is not None else getattr(reader.metadata, "nodata", None)

    for b in range(total_bands):
        band_data = reader.read_band(b)
        stats = calculate_band_statistics(band_data, nodata=effective_nodata, bins=bins)
        stats["band_index"] = b
        stats["band_name"] = (
            reader.metadata.band_details[b].name
            if reader.metadata.band_details and b < len(reader.metadata.band_details)
            else f"Band {b + 1}"
        )
        results.append(stats)
        if progress_callback:
            progress_callback(b + 1, total_bands)

    return results
