"""OpenENVI Color Space Transform Engine.

Implements standard remote sensing color space conversions:
- RGB to HSV (Hue, Saturation, Value)
- HSV to RGB
- RGB to Luminance / Grayscale
"""

from typing import List, Tuple
import numpy as np
import matplotlib.colors as mcolors


def normalize_to_unit(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize array values to [0.0, 1.0], returning min and max."""
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.float32), min_v, max_v
    norm = np.clip((arr - min_v) / (max_v - min_v), 0.0, 1.0).astype(np.float32)
    return norm, min_v, max_v


def rgb_to_hsv(
    r_band: np.ndarray,
    g_band: np.ndarray,
    b_band: np.ndarray,
    normalize_inputs: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Convert RGB bands to HSV color space.

    Args:
        r_band, g_band, b_band: 2D numpy arrays.
        normalize_inputs: If True, rescale inputs to [0, 1] range.

    Returns:
        Tuple of:
            - 3D array of shape (height, width, 3) where [..., 0]=Hue (0..1),
              [..., 1]=Saturation (0..1), [..., 2]=Value (0..1)
            - List of band names ['Hue', 'Saturation', 'Value']
    """
    if normalize_inputs:
        r_norm, _, _ = normalize_to_unit(r_band)
        g_norm, _, _ = normalize_to_unit(g_band)
        b_norm, _, _ = normalize_to_unit(b_band)
    else:
        r_norm = np.clip(r_band, 0.0, 1.0)
        g_norm = np.clip(g_band, 0.0, 1.0)
        b_norm = np.clip(b_band, 0.0, 1.0)

    rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
    hsv = mcolors.rgb_to_hsv(rgb)
    return hsv.astype(np.float32), ["Hue", "Saturation", "Value"]


def hsv_to_rgb(
    h_band: np.ndarray,
    s_band: np.ndarray,
    v_band: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Convert HSV bands back to RGB color space [0.0, 1.0]."""
    h_norm = np.clip(h_band, 0.0, 1.0)
    s_norm = np.clip(s_band, 0.0, 1.0)
    v_norm = np.clip(v_band, 0.0, 1.0)

    hsv = np.stack([h_norm, s_norm, v_norm], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb.astype(np.float32), ["Red", "Green", "Blue"]


def rgb_to_grayscale(
    r_band: np.ndarray,
    g_band: np.ndarray,
    b_band: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Convert RGB bands to standard luminance grayscale (Rec. 601)."""
    gray = 0.299 * r_band + 0.587 * g_band + 0.114 * b_band
    return gray[:, :, np.newaxis].astype(np.float32), ["Luminance_Grayscale"]
