"""OpenENVI Contrast Enhancement & Dynamic Stretch Engine.

Replicates ENVI's signature contrast stretch modes (Linear 2%, Linear 5%,
Histogram Equalization, Gaussian Stretch, and Min-Max normalization).
"""

from typing import Tuple
import numpy as np


def linear_percent_stretch(
    image: np.ndarray,
    percent: float = 2.0,
    nodata: float | None = None,
) -> np.ndarray:
    """Apply ENVI Linear Percentile Contrast Stretch (e.g. 2% or 5%).

    Clips the lowest `percent` and highest `percent` of pixel values and linearly
    scales the remaining dynamic range to [0, 1].

    Args:
        image: 2D or 3D numpy array.
        percent: Percentage to clip on each tail (e.g. 2.0 for 2% - 98%).
        nodata: Optional nodata value to mask out during stretch calculation.

    Returns:
        Stretched float32 image array with values in [0.0, 1.0].
    """
    if image.ndim == 3:
        # Multi-channel (e.g. RGB) - stretch each channel independently
        stretched_channels = [
            linear_percent_stretch(image[..., c], percent=percent, nodata=nodata)
            for c in range(image.shape[-1])
        ]
        return np.stack(stretched_channels, axis=-1)

    # 2D Grayscale array
    valid_mask = np.isfinite(image)
    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]
    if len(valid_pixels) == 0:
        return np.zeros_like(image, dtype=np.float32)

    # Fast sampling for large rasters (> 500k pixels) for instant rendering
    if len(valid_pixels) > 500_000:
        sample = valid_pixels[:: len(valid_pixels) // 250_000]
    else:
        sample = valid_pixels

    lower_val = float(np.percentile(sample, percent))
    upper_val = float(np.percentile(sample, 100.0 - percent))

    if upper_val <= lower_val:
        upper_val = lower_val + 1e-6

    stretched = (image.astype(np.float32) - lower_val) / (upper_val - lower_val)
    return np.clip(stretched, 0.0, 1.0)


def histogram_equalization_stretch(
    image: np.ndarray,
    num_bins: int = 256,
    nodata: float | None = None,
) -> np.ndarray:
    """Apply ENVI Histogram Equalization stretch.

    Redistributes pixel intensities uniformly across the dynamic range [0, 1].

    Args:
        image: 2D or 3D numpy array.
        num_bins: Number of histogram bins.
        nodata: Optional nodata value to ignore.

    Returns:
        Stretched float32 image array in [0.0, 1.0].
    """
    if image.ndim == 3:
        stretched_channels = [
            histogram_equalization_stretch(image[..., c], num_bins=num_bins, nodata=nodata)
            for c in range(image.shape[-1])
        ]
        return np.stack(stretched_channels, axis=-1)

    valid_mask = np.isfinite(image)
    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]
    if len(valid_pixels) == 0:
        return np.zeros_like(image, dtype=np.float32)

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
    stretched = np.interp(image.astype(np.float32), bin_centers, cdf).astype(np.float32)
    return np.clip(stretched, 0.0, 1.0)


def gaussian_stretch(
    image: np.ndarray,
    std_factor: float = 3.0,
    nodata: float | None = None,
) -> np.ndarray:
    """Apply ENVI Gaussian Stretch (mean +/- n * standard deviation).

    Args:
        image: 2D or 3D numpy array.
        std_factor: Multiplier for standard deviation (default 3.0).
        nodata: Optional nodata value.

    Returns:
        Stretched float32 image array in [0.0, 1.0].
    """
    if image.ndim == 3:
        stretched_channels = [
            gaussian_stretch(image[..., c], std_factor=std_factor, nodata=nodata)
            for c in range(image.shape[-1])
        ]
        return np.stack(stretched_channels, axis=-1)

    valid_mask = np.isfinite(image)
    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]
    if len(valid_pixels) == 0:
        return np.zeros_like(image, dtype=np.float32)

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

    stretched = (image.astype(np.float32) - lower_val) / (upper_val - lower_val)
    return np.clip(stretched, 0.0, 1.0)


def min_max_stretch(
    image: np.ndarray,
    nodata: float | None = None,
) -> np.ndarray:
    """Standard Min-Max normalization to [0.0, 1.0]."""
    if image.ndim == 3:
        stretched_channels = [
            min_max_stretch(image[..., c], nodata=nodata)
            for c in range(image.shape[-1])
        ]
        return np.stack(stretched_channels, axis=-1)

    valid_mask = np.isfinite(image)
    if nodata is not None:
        valid_mask &= (image != nodata)

    valid_pixels = image[valid_mask]
    if len(valid_pixels) == 0:
        return np.zeros_like(image, dtype=np.float32)

    min_val = float(np.min(valid_pixels))
    max_val = float(np.max(valid_pixels))

    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.float32)

    stretched = (image.astype(np.float32) - min_val) / (max_val - min_val)
    return np.clip(stretched, 0.0, 1.0)


def apply_stretch(
    image: np.ndarray,
    mode: str = "Linear 2%",
    nodata: float | None = None,
) -> np.ndarray:
    """Dispatcher function applying the requested contrast stretch mode.

    Args:
        image: 2D or 3D raw raster array.
        mode: Stretch mode identifier ('Linear 2%', 'Linear 5%', 'Equalization', 'Gaussian', 'No Stretch').
        nodata: Optional nodata value.

    Returns:
        Normalized display array of type uint8 (0-255).
    """
    if image.ndim == 3:
        # Sequential channel-by-channel processing directly into uint8.
        # Avoids allocating multiple simultaneous 3D float32 intermediate cubes (saves > 2 GB RAM on large rasters).
        h, w, c = image.shape
        out = np.empty((h, w, c), dtype=np.uint8)
        for i in range(c):
            out[..., i] = apply_stretch(image[..., i], mode=mode, nodata=nodata)
        return out

    mode_clean = mode.lower()
    if "2%" in mode_clean:
        norm = linear_percent_stretch(image, percent=2.0, nodata=nodata)
    elif "5%" in mode_clean:
        norm = linear_percent_stretch(image, percent=5.0, nodata=nodata)
    elif "equal" in mode_clean or "均衡" in mode_clean:
        norm = histogram_equalization_stretch(image, nodata=nodata)
    elif "gauss" in mode_clean or "高斯" in mode_clean:
        norm = gaussian_stretch(image, std_factor=3.0, nodata=nodata)
    else:
        norm = min_max_stretch(image, nodata=nodata)

    res = np.nan_to_num(norm * 255.0, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(res, 0.0, 255.0).astype(np.uint8)
