"""OpenENVI Reference Spectral Library.

Provides a built-in library of laboratory/field reference reflectance spectra so
the application can do what ENVI's Spectral Library Viewer does: browse known
materials, overlay them against a measured image spectrum, and rank the library
by spectral similarity without needing an external .slr file.

The spectra are generated from physically motivated models rather than stored as
tables. Vegetation is built from chlorophyll and water absorption features, soil
and rock from mineral absorption bands, and water and snow from explicit response
curves. Everything lands on one shared 1 nm wavelength grid from 400 to 2500 nm,
which is what makes cross-comparison between entries meaningful.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The single wavelength grid every library spectrum is defined on.
LIBRARY_WAVELENGTHS = np.arange(400.0, 2500.0 + 0.5, 1.0)


@dataclass
class LibrarySpectrum:
    """One reference spectrum on the shared wavelength grid."""

    name: str
    category: str
    reflectance: np.ndarray
    wavelengths: np.ndarray = field(default_factory=lambda: LIBRARY_WAVELENGTHS.copy())

    @property
    def peak_wavelength(self) -> float:
        """Wavelength of maximum reflectance, in nm."""
        return float(self.wavelengths[int(np.argmax(self.reflectance))])

    @property
    def mean_reflectance(self) -> float:
        """Average reflectance across the whole grid."""
        return float(np.mean(self.reflectance))


# ---------------------------------------------------------------- model helpers


def _gauss(wl: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Unit-height Gaussian band shape, used as an absorption feature."""
    return np.exp(-0.5 * ((wl - centre) / width) ** 2)


def _from_nodes(nodes: Sequence[Tuple[float, float]], wl: np.ndarray) -> np.ndarray:
    """Piecewise-linear curve through (wavelength, value) control points."""
    xs = np.array([n[0] for n in nodes], dtype=np.float64)
    ys = np.array([n[1] for n in nodes], dtype=np.float64)
    return np.interp(wl, xs, ys, left=ys[0], right=ys[-1])


def _vegetation(red_edge_nm: float, nir_level: float, dry: float = 0.0) -> np.ndarray:
    """Living vegetation curve: chlorophyll troughs, a red edge, NIR plateau.

    ``dry`` in [0, 1] blends towards senescent material by flattening the red
    edge and lifting the SWIR, which is how dry grass differs from green leaf.
    """
    wl = LIBRARY_WAVELENGTHS
    # Chlorophyll structure below the red edge: blue absorption, a green peak at
    # 550 nm, and the red trough that gives the edge its name.
    base = np.interp(
        wl,
        [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 690.0],
        [0.10, 0.06, 0.10, 0.30, 0.15, 0.07, 0.10],
    )
    # Logistic rise to the NIR plateau; the width sets how sharp the edge looks.
    edge = nir_level / (1.0 + np.exp(-(wl - red_edge_nm) / 22.0))
    curve = base + edge

    # Leaf water absorbs strongly near 1.4 and 1.9 um, which is what separates
    # vegetation from every other material in the library.
    curve *= 1.0 - 0.60 * _gauss(wl, 1400.0, 90.0)
    curve *= 1.0 - 0.62 * _gauss(wl, 1900.0, 80.0)
    curve *= 1.0 - 0.28 * _gauss(wl, 2300.0, 70.0)
    # Starch and cellulose give a shallow 2.1 um trough even in green material.
    curve *= 1.0 - 0.10 * _gauss(wl, 2100.0, 50.0)

    # Senescence: the NIR shoulder collapses and the SWIR lifts.
    curve = (1.0 - dry) * curve + dry * _from_nodes(
        [(400, 0.14), (700, 0.26), (1100, 0.30), (1400, 0.22), (1900, 0.16), (2500, 0.20)],
        wl,
    )
    return np.clip(curve, 0.0, None)


def _soil(base: float, features: Sequence[Tuple[float, float, float]], slope: float) -> np.ndarray:
    """Soil/rock curve: a bright base with mineral absorption bands.

    ``features`` are (centre_nm, width_nm, depth) absorption bands and ``slope``
    is the overall brightening with wavelength typical of dry mineral surfaces.
    """
    wl = LIBRARY_WAVELENGTHS
    curve = base * (1.0 + slope * ((wl - 400.0) / 2100.0))
    for centre, width, depth in features:
        curve *= 1.0 - depth * _gauss(wl, centre, width)
    return np.clip(curve, 0.0, None)


# ------------------------------------------------------------------- the library


def _build_library() -> List[LibrarySpectrum]:
    """Construct every built-in reference spectrum."""
    entries: List[Tuple[str, str, np.ndarray]] = [
        # Vegetation
        ("Green Vegetation", "Vegetation", _vegetation(700.0, 0.52, dry=0.0)),
        ("Mature Leaf", "Vegetation", _vegetation(720.0, 0.58, dry=0.0)),
        ("Conifer Needles", "Vegetation", _vegetation(760.0, 0.40, dry=0.0)),
        ("Dry Grass", "Vegetation", _vegetation(700.0, 0.42, dry=0.75)),
        ("Senescent Foliage", "Vegetation", _vegetation(700.0, 0.40, dry=1.0)),

        # Soil
        ("Dry Sand", "Soil", _soil(0.30, [(1400.0, 70.0, 0.15), (1900.0, 90.0, 0.22)], 0.25)),
        ("Loam", "Soil", _soil(0.24, [(1400.0, 80.0, 0.28), (1900.0, 95.0, 0.36)], 0.15)),
        ("Wet Soil", "Soil", _soil(0.16, [(1400.0, 80.0, 0.45), (1900.0, 95.0, 0.55)], 0.10)),
        ("Red Soil (Ferric)", "Soil", _soil(0.27, [(900.0, 110.0, 0.30), (1900.0, 90.0, 0.30)], 0.20)),

        # Rock and minerals
        ("Granite", "Rock", _soil(0.32, [(1400.0, 75.0, 0.12), (2200.0, 80.0, 0.15)], 0.18)),
        ("Basalt", "Rock", _soil(0.18, [(1400.0, 75.0, 0.14), (2200.0, 80.0, 0.18)], 0.12)),
        ("Limestone", "Rock", _soil(0.45, [(1400.0, 90.0, 0.10), (1900.0, 100.0, 0.30), (2300.0, 70.0, 0.22)], 0.10)),
        ("Dolomite", "Rock", _soil(0.38, [(1400.0, 90.0, 0.12), (1900.0, 100.0, 0.18), (2300.0, 70.0, 0.15)], 0.12)),
        ("Hematite", "Rock", _soil(0.18, [(880.0, 60.0, 0.45), (660.0, 50.0, 0.25), (1900.0, 90.0, 0.25)], 0.05)),
        ("Kaolinite", "Rock", _soil(0.42, [(1400.0, 70.0, 0.32), (2200.0, 60.0, 0.40)], 0.10)),
        ("Calcite", "Rock", _soil(0.48, [(1400.0, 80.0, 0.14), (1900.0, 95.0, 0.28), (2350.0, 45.0, 0.45)], 0.08)),

        # Water
        ("Clear Water", "Water", _from_nodes(
            [(400, 0.045), (600, 0.032), (700, 0.012), (900, 0.004),
             (1200, 0.001), (2500, 0.0)],
            LIBRARY_WAVELENGTHS,
        )),
        ("Turbid Water", "Water", _from_nodes(
            [(400, 0.115), (600, 0.095), (700, 0.072), (800, 0.058),
             (1000, 0.040), (1400, 0.020), (1900, 0.010), (2500, 0.005)],
            LIBRARY_WAVELENGTHS,
        )),

        # Snow and ice
        ("Dry Snow", "Snow/Ice", _from_nodes(
            [(400, 0.80), (600, 0.86), (800, 0.88), (1100, 0.87),
             (1400, 0.62), (1500, 0.80), (1900, 0.45), (2100, 0.72), (2500, 0.55)],
            LIBRARY_WAVELENGTHS,
        )),
        ("Ice", "Snow/Ice", _from_nodes(
            [(400, 0.58), (600, 0.62), (800, 0.60), (1100, 0.55),
             (1400, 0.34), (1500, 0.50), (1900, 0.22), (2100, 0.44), (2500, 0.30)],
            LIBRARY_WAVELENGTHS,
        )),
    ]
    return [
        LibrarySpectrum(name=name, category=category, reflectance=np.asarray(values, np.float64))
        for name, category, values in entries
    ]


_LIBRARY: Optional[List[LibrarySpectrum]] = None


def get_spectral_library() -> List[LibrarySpectrum]:
    """Return the built-in reference spectra, built once and cached."""
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = _build_library()
    return list(_LIBRARY)


def get_categories() -> List[str]:
    """Category names in the order the library presents them."""
    seen: List[str] = []
    for entry in get_spectral_library():
        if entry.category not in seen:
            seen.append(entry.category)
    return seen


def find_spectrum(name: str) -> Optional[LibrarySpectrum]:
    """Look up one library entry by its exact name."""
    for entry in get_spectral_library():
        if entry.name == name:
            return entry
    return None


# ------------------------------------------------------------------- operations


def resample_spectrum(
    wavelengths: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Interpolate a measured spectrum onto the library's wavelength grid.

    Samples outside the measured range become NaN instead of being clamped to
    the nearest value, so a narrow-band sensor is never compared against
    wavelengths it never observed.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)

    order = np.argsort(wl)
    wl_sorted = wl[order]
    vals_sorted = vals[order]
    # A duplicated wavelength would make np.interp undefined on that point.
    unique_wl, unique_idx = np.unique(wl_sorted, return_index=True)

    out = np.interp(tgt, unique_wl, vals_sorted[unique_idx], left=np.nan, right=np.nan)
    return out


def mean_spectrum(
    cube: np.ndarray,
    wavelengths: Sequence[float],
) -> Tuple[np.ndarray, int]:
    """Average spectrum over every valid pixel of a (bands, lines, samples) cube.

    Returns the (bands,) mean curve and the number of pixels that contributed.
    """
    data = np.asarray(cube, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"Expected a (bands, lines, samples) cube; got shape {data.shape}.")
    wl = np.asarray(wavelengths, dtype=np.float64)
    if data.shape[0] != wl.size:
        raise ValueError(
            f"Cube has {data.shape[0]} bands but {wl.size} wavelengths were supplied."
        )

    valid = np.all(np.isfinite(data), axis=0)
    count = int(valid.sum())
    if count == 0:
        raise ValueError("Cube contains no valid pixels to average.")

    stacked = data[:, valid].T           # (pixels, bands)
    return np.nanmean(stacked, axis=0), count


def spectral_angle_degrees(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    """Spectral angle in degrees between two equal-length curves.

    The angle is scale invariant, so a brightness mismatch does not penalise a
    match; only the shape matters, which is what makes it usable across sensors.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    a = np.where(finite, a, 0.0)
    b = np.where(finite, b, 0.0)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return float("nan")
    cos = float(np.dot(a, b) / denom)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def match_library(
    query_wavelengths: np.ndarray,
    query_values: np.ndarray,
    max_angle: Optional[float] = None,
    categories: Optional[Sequence[str]] = None,
) -> List[Tuple[LibrarySpectrum, float]]:
    """Rank library spectra by spectral angle against a measured spectrum.

    Args:
        query_wavelengths: Wavelengths of the measured spectrum.
        query_values: The measured reflectance curve.
        max_angle: Optional cutoff in degrees; matches beyond it are dropped.
        categories: Optional category whitelist.

    Returns:
        List of (entry, angle_degrees) sorted by increasing angle.
    """
    query = resample_spectrum(query_wavelengths, query_values, LIBRARY_WAVELENGTHS)
    if not np.isfinite(query).any():
        raise ValueError(
            "Measured spectrum does not overlap the library range "
            f"{LIBRARY_WAVELENGTHS[0]:.0f}-{LIBRARY_WAVELENGTHS[-1]:.0f} nm."
        )

    allowed = set(categories) if categories else None
    results: List[Tuple[LibrarySpectrum, float]] = []
    for entry in get_spectral_library():
        if allowed is not None and entry.category not in allowed:
            continue
        angle = spectral_angle_degrees(query, entry.reflectance)
        if not np.isfinite(angle):
            continue
        if max_angle is not None and angle > float(max_angle):
            continue
        results.append((entry, angle))
    results.sort(key=lambda item: item[1])
    return results