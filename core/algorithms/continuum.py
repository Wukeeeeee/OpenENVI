"""OpenENVI Continuum Removal (CR) Engine.

Normalizes spectral curves by fitting an upper convex hull over local maxima
and dividing reflectance by the continuum line: R_cr(lambda) = R(lambda) / C(lambda).
Crucial for quantitative hyperspectral mineralogy and diagnostic absorption analysis.
"""

from typing import Callable, Optional, Tuple
import numpy as np


def continuum_removal_1d(
    spectrum: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute continuum-removed spectrum and continuum hull for a 1D spectral curve.

    Args:
        spectrum: 1D array of reflectance or radiance values.
        wavelengths: Optional 1D array of corresponding wavelengths.

    Returns:
        Tuple of:
            - continuum_removed: 1D float32 array in [0.0, 1.0]
            - continuum_hull: 1D float32 array of interpolated convex hull values
    """
    n = len(spectrum)
    if n < 3:
        return spectrum.astype(np.float32), spectrum.astype(np.float32)

    if wavelengths is None:
        wl = np.arange(n, dtype=np.float64)
    else:
        wl = np.asarray(wavelengths, dtype=np.float64)

    y = np.asarray(spectrum, dtype=np.float64)

    # Monotone chain upper convex hull
    pts = np.column_stack((wl, y))
    upper = [pts[0]]
    for p in pts[1:]:
        while len(upper) >= 2:
            p1, p2 = upper[-2], upper[-1]
            # Cross product (p2 - p1) x (p - p1)
            # Negative cross product indicates clockwise (right turn) for upper hull
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross >= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    upper_arr = np.array(upper)

    # Linear interpolation across wavelengths
    continuum = np.interp(wl, upper_arr[:, 0], upper_arr[:, 1])
    continuum = np.maximum(continuum, 1e-7)

    cr = np.clip(y / continuum, 0.0, 1.0)
    return cr.astype(np.float32), continuum.astype(np.float32)


def continuum_removal_cube(
    cube: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Compute continuum removal for an entire 3D raster cube (height, width, bands).

    Args:
        cube: 3D numpy array of shape (height, width, bands).
        wavelengths: Optional 1D array of wavelengths.
        progress_callback: Optional (current_line, total_lines) callback.

    Returns:
        3D float32 array of shape (height, width, bands) with continuum removed.
    """
    h, w, b = cube.shape
    out = np.empty_like(cube, dtype=np.float32)

    if wavelengths is None:
        wl = np.arange(b, dtype=np.float64)
    else:
        wl = np.asarray(wavelengths, dtype=np.float64)

    for y in range(h):
        for x in range(w):
            cr, _ = continuum_removal_1d(cube[y, x, :], wavelengths=wl)
            out[y, x, :] = cr

        if progress_callback:
            progress_callback(y + 1, h)

    return out
