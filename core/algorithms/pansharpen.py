"""OpenENVI Pan-Sharpening (High-Resolution Image Fusion) Engine.

Provides Gram-Schmidt (GS) and Brovey fusion algorithms to sharpen lower-resolution
multispectral (e.g., 30m Landsat MS) observations with a higher-resolution
panchromatic band (e.g., 15m Landsat Band 8) to synthesize high-resolution color imagery.
"""

from typing import Callable, List, Optional, Union
import numpy as np
import scipy.ndimage


def resample_band_to_grid(
    band_data: np.ndarray,
    target_height: int,
    target_width: int,
    order: int = 1,
) -> np.ndarray:
    """Resample a 2D band to target spatial dimensions using bilinear interpolation.

    Args:
        band_data: 2D numpy array of shape (H_in, W_in).
        target_height: Target line count.
        target_width: Target sample count.
        order: Spline interpolation order (0: nearest-neighbor, 1: bilinear, 3: bicubic).

    Returns:
        2D numpy array of shape (target_height, target_width).
    """
    in_h, in_w = band_data.shape[:2]
    if in_h == target_height and in_w == target_width:
        return band_data.astype(np.float32, copy=False)

    zoom_y = target_height / float(in_h)
    zoom_x = target_width / float(in_w)

    resampled = scipy.ndimage.zoom(band_data, (zoom_y, zoom_x), order=order)

    # Ensure exact dimensions matching target grid
    curr_h, curr_w = resampled.shape[:2]
    if curr_h != target_height or curr_w != target_width:
        out = np.zeros((target_height, target_width), dtype=np.float32)
        h_crop = min(curr_h, target_height)
        w_crop = min(curr_w, target_width)
        out[:h_crop, :w_crop] = resampled[:h_crop, :w_crop]
        return out

    return resampled.astype(np.float32, copy=False)


def brovey_pansharpen(
    pan_band: np.ndarray,
    ms_bands: List[np.ndarray],
    weights: Optional[List[float]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Perform Brovey transform pan-sharpening.

    Formula:
        I = sum(w_i * MS_i)
        MS_sharp_i = MS_i * (Pan / (I + epsilon))

    Args:
        pan_band: 2D numpy array of shape (H_pan, W_pan).
        ms_bands: List of 2D numpy arrays [MS_1, MS_2, ...].
        weights: Optional spectral contribution weights for each band.
        progress_callback: Optional callback receiving (step, total_steps).

    Returns:
        3D numpy array of shape (H_pan, W_pan, len(ms_bands)) with fused high-res bands.
    """
    target_h, target_w = pan_band.shape[:2]
    n_bands = len(ms_bands)
    if n_bands == 0:
        raise ValueError("Must provide at least one multispectral band for Pan-Sharpening.")

    total_steps = n_bands * 2 + 2
    step = 0

    if weights is None:
        weights = [1.0 / n_bands] * n_bands
    else:
        total_w = sum(weights) or 1.0
        weights = [w / total_w for w in weights]

    # 1. Resample each MS band to Pan grid and accumulate synthetic intensity I
    resampled_ms: List[np.ndarray] = []
    intensity = np.zeros((target_h, target_w), dtype=np.float32)

    for i, b in enumerate(ms_bands):
        res_b = resample_band_to_grid(b, target_h, target_w, order=1)
        resampled_ms.append(res_b)
        intensity += weights[i] * res_b
        step += 1
        if progress_callback:
            progress_callback(step, total_steps)

    # 2. Compute ratio Pan / (Intensity + eps)
    eps = 1e-6
    ratio = pan_band.astype(np.float32) / (intensity + eps)
    step += 1
    if progress_callback:
        progress_callback(step, total_steps)

    # 3. Apply ratio to sharpen each band
    sharpened = np.zeros((target_h, target_w, n_bands), dtype=np.float32)
    for i, res_b in enumerate(resampled_ms):
        sharpened[..., i] = res_b * ratio
        step += 1
        if progress_callback:
            progress_callback(step, total_steps)

    return sharpened


def gram_schmidt_pansharpen(
    pan_band: np.ndarray,
    ms_bands: List[np.ndarray],
    weights: Optional[List[float]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Perform Gram-Schmidt (GS) orthogonalization pan-sharpening.

    Matches the statistical histogram of Pan to the simulated GS1 band,
    orthogonalizes low-res components, and projects high-frequency spatial details.

    Args:
        pan_band: 2D numpy array of shape (H_pan, W_pan).
        ms_bands: List of 2D numpy arrays [MS_1, MS_2, ...].
        weights: Optional weights to simulate the low-res panchromatic band.
        progress_callback: Optional callback receiving (step, total_steps).

    Returns:
        3D numpy array of shape (H_pan, W_pan, len(ms_bands)).
    """
    target_h, target_w = pan_band.shape[:2]
    n_bands = len(ms_bands)
    if n_bands == 0:
        raise ValueError("Must provide at least one multispectral band for Pan-Sharpening.")

    total_steps = n_bands * 2 + 2
    step = 0

    if weights is None:
        weights = [1.0 / n_bands] * n_bands
    else:
        total_w = sum(weights) or 1.0
        weights = [w / total_w for w in weights]

    # 1. Resample MS bands to Pan dimensions and simulate GS1
    resampled_ms: List[np.ndarray] = []
    gs1 = np.zeros((target_h, target_w), dtype=np.float32)

    for i, b in enumerate(ms_bands):
        res_b = resample_band_to_grid(b, target_h, target_w, order=1)
        resampled_ms.append(res_b)
        gs1 += weights[i] * res_b
        step += 1
        if progress_callback:
            progress_callback(step, total_steps)

    # 2. Histogram match Pan to GS1
    # Use subsampled statistics for high speed on multi-gigabyte rasters
    sample_step = max(1, max(target_h, target_w) // 500)
    pan_sub = pan_band[::sample_step, ::sample_step].astype(np.float32)
    gs1_sub = gs1[::sample_step, ::sample_step]

    mu_pan = float(np.mean(pan_sub))
    std_pan = float(np.std(pan_sub)) + 1e-6

    mu_gs1 = float(np.mean(gs1_sub))
    std_gs1 = float(np.std(gs1_sub))

    # Matched Pan: (Pan - mu_pan) / std_pan * std_gs1 + mu_gs1
    # Compute difference: delta = pan_matched - gs1
    matched_pan = (pan_band.astype(np.float32) - mu_pan) / std_pan * std_gs1 + mu_gs1
    delta = matched_pan - gs1
    del matched_pan

    step += 1
    if progress_callback:
        progress_callback(step, total_steps)

    # 3. Compute Gram-Schmidt projection coefficients and apply spatial detail
    var_gs1 = float(np.var(gs1_sub)) + 1e-6
    sharpened = np.zeros((target_h, target_w, n_bands), dtype=np.float32)

    for i, res_b in enumerate(resampled_ms):
        res_b_sub = res_b[::sample_step, ::sample_step]
        cov_val = float(np.cov(res_b_sub.ravel(), gs1_sub.ravel())[0, 1])
        alpha = cov_val / var_gs1

        sharpened[..., i] = res_b + alpha * delta
        step += 1
        if progress_callback:
            progress_callback(step, total_steps)

    return sharpened
