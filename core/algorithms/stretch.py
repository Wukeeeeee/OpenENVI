"""OpenENVI Contrast Enhancement & Dynamic Stretch Engine.

Replicates ENVI's signature contrast stretch modes (Linear 2%, Linear 5%,
Histogram Equalization, Gaussian Stretch, and Min-Max normalization).

Memory optimisation notes
--------------------------
All per-channel stretch helpers accept an optional ``out`` parameter so the
caller can supply a pre-allocated float32 work buffer and avoid creating a new
array on every call.  The ``apply_stretch`` dispatcher reuses a single
(H, W) buffer across all channels of an RGB image, cutting peak working memory
from ~3 × band_size to ~1 × band_size during colour-composite display.
"""

from typing import Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Per-channel stretch helpers
# ---------------------------------------------------------------------------

def linear_percent_stretch(
    image: np.ndarray,
    percent: float = 2.0,
    nodata: float | None = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply ENVI Linear Percentile Contrast Stretch (e.g. 2% or 5%).

    Clips the lowest ``percent`` and highest ``percent`` of pixel values and
    linearly scales the remaining dynamic range to [0, 1].

    Args:
        image: 2D numpy array (single channel).
        percent: Percentage to clip on each tail (e.g. 2.0 for 2%–98%).
        nodata: Optional nodata value to mask out during stretch calculation.
        out: Optional pre-allocated float32 2D output array (same shape as
             ``image``).  When supplied, the result is written in-place to
             avoid an extra allocation.

    Returns:
        Stretched float32 array with values in [0.0, 1.0].  If ``out`` was
        provided, the same object is returned.
    """
    valid_mask = np.isfinite(image)
    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]
    if out is None:
        out = np.empty_like(image, dtype=np.float32)

    if len(valid_pixels) == 0:
        out[:] = 0.0
        return out

    # Fast sampling for large rasters (>500k pixels) for instant rendering
    if len(valid_pixels) > 500_000:
        sample = valid_pixels[:: len(valid_pixels) // 250_000]
    else:
        sample = valid_pixels

    lower_val = float(np.percentile(sample, percent))
    upper_val = float(np.percentile(sample, 100.0 - percent))

    if upper_val <= lower_val:
        upper_val = lower_val + 1e-6

    # In-place arithmetic: out = (image - lower) / (upper - lower)
    np.subtract(image, lower_val, out=out, casting="unsafe")
    out /= upper_val - lower_val
    np.clip(out, 0.0, 1.0, out=out)
    out[~valid_mask] = 0.0
    return out


def histogram_equalization_stretch(
    image: np.ndarray,
    num_bins: int = 256,
    nodata: float | None = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply ENVI Histogram Equalization stretch.

    Redistributes pixel intensities uniformly across the dynamic range [0, 1].

    Args:
        image: 2D numpy array (single channel).
        num_bins: Number of histogram bins.
        nodata: Optional nodata value to ignore.
        out: Optional pre-allocated float32 output array.

    Returns:
        Stretched float32 array in [0.0, 1.0].
    """
    valid_mask = np.isfinite(image)
    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]
    if out is None:
        out = np.empty_like(image, dtype=np.float32)

    if len(valid_pixels) == 0:
        out[:] = 0.0
        return out

    if len(valid_pixels) > 500_000:
        sample = valid_pixels[:: len(valid_pixels) // 250_000]
    else:
        sample = valid_pixels

    hist, bin_edges = np.histogram(sample, bins=num_bins)
    cdf = hist.cumsum().astype(np.float64)
    if cdf[-1] > 0:
        cdf /= cdf[-1]  # Normalize CDF to [0, 1]

    # Map original pixel values through the CDF lookup table
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    result = np.interp(image.astype(np.float32), bin_centers, cdf).astype(np.float32)
    np.clip(result, 0.0, 1.0, out=out)
    out[~valid_mask] = 0.0
    return out


def gaussian_stretch(
    image: np.ndarray,
    std_factor: float = 3.0,
    nodata: float | None = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply ENVI Gaussian Stretch (mean ± n × standard deviation).

    Args:
        image: 2D numpy array (single channel).
        std_factor: Multiplier for standard deviation (default 3.0).
        nodata: Optional nodata value.
        out: Optional pre-allocated float32 output array.

    Returns:
        Stretched float32 array in [0.0, 1.0].
    """
    valid_mask = np.isfinite(image)
    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]
    if out is None:
        out = np.empty_like(image, dtype=np.float32)

    if len(valid_pixels) == 0:
        out[:] = 0.0
        return out

    if len(valid_pixels) > 500_000:
        sample = valid_pixels[:: len(valid_pixels) // 250_000]
    else:
        sample = valid_pixels

    mean_val = float(np.mean(sample))
    std_val = float(np.std(sample))

    lower_val = mean_val - std_factor * std_val
    upper_val = mean_val + std_factor * std_val

    if upper_val <= lower_val:
        upper_val = lower_val + 1e-6

    np.subtract(image, lower_val, out=out, casting="unsafe")
    out /= upper_val - lower_val
    np.clip(out, 0.0, 1.0, out=out)
    out[~valid_mask] = 0.0
    return out


def min_max_stretch(
    image: np.ndarray,
    nodata: float | None = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Standard Min-Max normalization to [0.0, 1.0].

    Args:
        image: 2D numpy array (single channel).
        nodata: Optional nodata value.
        out: Optional pre-allocated float32 output array.

    Returns:
        Normalized float32 array in [0.0, 1.0].
    """
    valid_mask = np.isfinite(image)
    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]
    if out is None:
        out = np.empty_like(image, dtype=np.float32)

    if len(valid_pixels) == 0:
        out[:] = 0.0
        return out

    min_val = float(np.min(valid_pixels))
    max_val = float(np.max(valid_pixels))

    if max_val <= min_val:
        out[:] = 0.0
        return out

    np.subtract(image, min_val, out=out, casting="unsafe")
    out /= max_val - min_val
    np.clip(out, 0.0, 1.0, out=out)
    out[~valid_mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def apply_stretch(
    image: np.ndarray,
    mode: str = "Linear 2%",
    nodata: float | None = None,
) -> np.ndarray:
    """Apply the requested contrast stretch mode and return a display-ready uint8 array.

    Memory optimisation
    -------------------
    For 3-D (H, W, C) inputs a single float32 work buffer ``chan_buf`` is
    allocated once and reused across all channels.  This replaces the previous
    approach that created a new array per channel and then stacked them,
    which could hold up to 3 × channel_size of temporary float32 data
    simultaneously.  Peak working memory for a 3-channel image is now
    ~1 × channel_size instead of ~3 × channel_size (saves ≈ 400–450 MB on a
    full 7700 × 7500 Landsat scene).

    Args:
        image: 2D or 3D raw raster array.
        mode: Stretch mode identifier ('Linear 2%', 'Linear 5%',
              'Equalization', 'Gaussian', 'No Stretch').
        nodata: Optional nodata value.

    Returns:
        Normalized display array of type uint8 (0-255).
    """
    if image.ndim == 3:
        h, w, c = image.shape
        out = np.empty((h, w, c), dtype=np.uint8)
        # Single reusable float32 work buffer – shared across all channels.
        chan_buf = np.empty((h, w), dtype=np.float32)
        for i in range(c):
            _apply_stretch_2d(image[..., i], mode=mode, nodata=nodata, buf=chan_buf)
            # Convert [0,1] float32 → uint8 in-place using the same buffer
            np.multiply(chan_buf, 255.0, out=chan_buf)
            np.nan_to_num(chan_buf, nan=0.0, posinf=255.0, neginf=0.0, copy=False)
            np.clip(chan_buf, 0.0, 255.0, out=chan_buf)
            out[..., i] = chan_buf  # implicit uint8 cast (no extra alloc)
        return out

    # 2-D grayscale path
    norm = _apply_stretch_2d(image, mode=mode, nodata=nodata)
    res = np.nan_to_num(norm * 255.0, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(res, 0.0, 255.0).astype(np.uint8)


def _apply_stretch_2d(
    image: np.ndarray,
    mode: str,
    nodata: float | None,
    buf: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Internal helper: apply stretch to a single 2-D channel.

    Args:
        image: 2D numpy array.
        mode: Stretch mode string.
        nodata: Optional nodata value.
        buf: Optional pre-allocated float32 work buffer (same shape as image).
             When provided, the result is written into ``buf`` and returned.

    Returns:
        Stretched float32 array in [0.0, 1.0] (may be the same object as ``buf``).
    """
    mode_clean = mode.lower()
    if "2%" in mode_clean:
        return linear_percent_stretch(image, percent=2.0, nodata=nodata, out=buf)
    elif "5%" in mode_clean:
        return linear_percent_stretch(image, percent=5.0, nodata=nodata, out=buf)
    elif "equal" in mode_clean or "均衡" in mode_clean:
        return histogram_equalization_stretch(image, nodata=nodata, out=buf)
    elif "gauss" in mode_clean or "高斯" in mode_clean:
        return gaussian_stretch(image, std_factor=3.0, nodata=nodata, out=buf)
    else:
        return min_max_stretch(image, nodata=nodata, out=buf)
