"""OpenENVI Synthetic Hyperspectral Benchmark Generator.

Generates realistic multi-endmember hyperspectral image cubes (vegetation, soil,
water, urban) with Gaussian absorption features, georeferencing, and ENVI / GeoTIFF export.
"""

import os
from typing import List, Optional, Tuple
import numpy as np

import core.proj_setup  # Configure PROJ paths before importing rasterio
import rasterio
from rasterio.transform import from_origin


def generate_synthetic_spectra(wavelengths: np.ndarray) -> dict[str, np.ndarray]:
    """Generate realistic spectral reflectance signatures for 4 distinct endmembers.

    Args:
        wavelengths: 1D array of wavelengths in nanometers (e.g. 400 - 2400 nm).

    Returns:
        Dictionary mapping class names to reflectance arrays.
    """
    wl = wavelengths.astype(np.float64)

    # 1. Vegetation: Low in Blue/Red, Green peak (~550nm), steep Red-Edge (~700-750nm), high NIR plateau, water absorptions
    veg = np.zeros_like(wl)
    veg += 0.05  # baseline
    veg += 0.08 * np.exp(-((wl - 550) ** 2) / (2 * 40**2))  # green peak
    # Red edge sigmoid transition
    veg += 0.45 / (1.0 + np.exp(-(wl - 720) / 25.0))
    # Water vapor absorption valleys at ~970, 1200, 1400, 1940 nm
    for valley, width, depth in [(970, 40, 0.1), (1200, 50, 0.15), (1400, 60, 0.35), (1940, 70, 0.4)]:
        veg -= depth * np.exp(-((wl - valley) ** 2) / (2 * width**2))
    veg = np.clip(veg, 0.01, 0.85)

    # 2. Water: High in Blue-Green, rapidly dropping to near zero in NIR/SWIR
    water = 0.12 * np.exp(-((wl - 480) ** 2) / (2 * 120**2))
    water[wl > 750] = 0.01
    water = np.clip(water, 0.005, 0.2)

    # 3. Soil: Smooth upward curve from visible to SWIR
    soil = 0.10 + 0.35 * (wl - 400) / (2400 - 400)
    # Clay absorption feature at 2200 nm
    soil -= 0.08 * np.exp(-((wl - 2200) ** 2) / (2 * 30**2))
    soil = np.clip(soil, 0.05, 0.6)

    # 4. Urban / Concrete: Relatively flat, moderately high reflectance
    urban = 0.25 + 0.10 * np.sin((wl - 400) / 400.0)
    urban = np.clip(urban, 0.15, 0.5)

    return {
        "vegetation": veg.astype(np.float32),
        "water": water.astype(np.float32),
        "soil": soil.astype(np.float32),
        "urban": urban.astype(np.float32),
    }


def generate_synthetic_cube(
    lines: int = 128,
    samples: int = 128,
    bands: int = 64,
    start_wl: float = 400.0,
    end_wl: float = 2400.0,
    noise_std: float = 0.005,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic hyperspectral cube with distinct spatial zones.

    Args:
        lines: Height (rows).
        samples: Width (columns).
        bands: Number of spectral bands.
        start_wl: Start wavelength (nm).
        end_wl: End wavelength (nm).
        noise_std: Standard deviation of Gaussian sensor noise.
        seed: Random seed for repeatability.

    Returns:
        Tuple of (cube, wavelengths, ground_truth_classes)
        - cube: np.ndarray shape (bands, lines, samples)
        - wavelengths: np.ndarray shape (bands,)
        - ground_truth: np.ndarray shape (lines, samples) with class indices (0:veg, 1:water, 2:soil, 3:urban)
    """
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(start_wl, end_wl, bands, dtype=np.float32)
    endmembers = generate_synthetic_spectra(wavelengths)

    em_veg = endmembers["vegetation"]
    em_water = endmembers["water"]
    em_soil = endmembers["soil"]
    em_urban = endmembers["urban"]

    gt = np.zeros((lines, samples), dtype=np.int32)
    cube = np.zeros((bands, lines, samples), dtype=np.float32)

    # Divide into 4 quadrants / spatial zones with smooth mixing at boundaries
    half_l = lines // 2
    half_s = samples // 2

    # Top-Left: Vegetation
    gt[:half_l, :half_s] = 0
    # Top-Right: Water
    gt[:half_l, half_s:] = 1
    # Bottom-Left: Soil
    gt[half_l:, :half_s] = 2
    # Bottom-Right: Urban
    gt[half_l:, half_s:] = 3

    for b in range(bands):
        cube[b, :half_l, :half_s] = em_veg[b]
        cube[b, :half_l, half_s:] = em_water[b]
        cube[b, half_l:, :half_s] = em_soil[b]
        cube[b, half_l:, half_s:] = em_urban[b]

    # Add Gaussian sensor noise
    noise = rng.normal(0, noise_std, cube.shape).astype(np.float32)
    cube = np.clip(cube + noise, 0.0, 1.0)

    return cube, wavelengths, gt


def write_envi_dataset(
    output_base_path: str,
    cube: np.ndarray,
    wavelengths: np.ndarray,
    interleave: str = "bsq",
    easting: float = 500000.0,
    northing: float = 3500000.0,
    pixel_size: float = 10.0,
) -> Tuple[str, str]:
    """Write a hyperspectral cube to ENVI .hdr and .dat files.

    Args:
        output_base_path: Base file path without extension (e.g. '/path/to/sample_cube').
        cube: 3D numpy array of shape (bands, lines, samples).
        wavelengths: 1D array of wavelengths.
        interleave: 'bsq', 'bil', or 'bip'.
        easting: Tie point easting coordinate.
        northing: Tie point northing coordinate.
        pixel_size: Pixel spatial resolution in meters.

    Returns:
        Tuple of (hdr_path, dat_path).
    """
    bands, lines, samples = cube.shape
    hdr_path = output_base_path + ".hdr"
    dat_path = output_base_path + ".dat"

    # Transpose data according to interleave format
    interleave = interleave.lower()
    if interleave == "bsq":
        out_cube = np.ascontiguousarray(cube, dtype=np.float32)
    elif interleave == "bil":
        # (bands, lines, samples) -> (lines, bands, samples)
        out_cube = np.ascontiguousarray(np.transpose(cube, (1, 0, 2)), dtype=np.float32)
    elif interleave == "bip":
        # (bands, lines, samples) -> (lines, samples, bands)
        out_cube = np.ascontiguousarray(np.transpose(cube, (1, 2, 0)), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported interleave format: {interleave}")

    # Write raw binary data file
    with open(dat_path, "wb") as f:
        out_cube.tofile(f)

    # Format wavelengths string
    wl_str = ", ".join(f"{w:.2f}" for w in wavelengths)

    # Write ENVI .hdr file
    header_content = f"""ENVI
description = {{OpenENVI Synthetic Hyperspectral Benchmark Dataset}}
samples = {samples}
lines = {lines}
bands = {bands}
header offset = 0
file type = ENVI Standard
data type = 4
interleave = {interleave}
sensor type = Unknown
byte order = 0
wavelength units = nm
map info = {{UTM, 1.000, 1.000, {easting:.3f}, {northing:.3f}, {pixel_size:.6f}, {-pixel_size:.6f}, 50, North, WGS-84, units=Meters}}
wavelength = {{{wl_str}}}
"""
    with open(hdr_path, "w", encoding="utf-8") as f:
        f.write(header_content)

    return hdr_path, dat_path


def write_geotiff_dataset(
    output_path: str,
    cube: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    easting: float = 500000.0,
    northing: float = 3500000.0,
    pixel_size: float = 10.0,
    epsg: int = 32650,
) -> str:
    """Write a multispectral / hyperspectral cube to GeoTIFF format.

    Args:
        output_path: Path to .tif file.
        cube: 3D numpy array of shape (bands, lines, samples).
        wavelengths: Optional 1D array of wavelengths.
        easting: Top-left X coordinate.
        northing: Top-left Y coordinate.
        pixel_size: Pixel spatial resolution in meters.
        epsg: CRS EPSG code (e.g. 32650 for UTM 50N).

    Returns:
        output_path
    """
    bands, lines, samples = cube.shape
    transform = from_origin(easting, northing, pixel_size, pixel_size)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=lines,
        width=samples,
        count=bands,
        dtype=cube.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
    ) as dst:
        for b in range(bands):
            dst.write(cube[b], b + 1)
            if wavelengths is not None and b < len(wavelengths):
                dst.set_band_description(b + 1, f"Band {b + 1} ({wavelengths[b]:.1f} nm)")

    return output_path
