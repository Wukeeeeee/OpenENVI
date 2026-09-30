"""OpenENVI Radiometric Calibration & Atmospheric Correction Engine.

Provides quantitative conversion of raw satellite Digital Numbers (DN) to
Top-of-Atmosphere (TOA) Spectral Radiance, TOA Planetary Reflectance, and
Surface Reflectance via Dark Object Subtraction (DOS-1).
Specifically optimized for USGS Landsat 8/9 and Landsat 4/5/7 MTL metadata.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from core.io.base import BaseRasterReader


def extract_landsat_cal_params(
    reader: BaseRasterReader,
) -> Optional[Dict[str, Any]]:
    """Extract radiometric rescaling factors and sun geometry from reader metadata.

    Returns:
        Dictionary with 'sun_elevation', 'reflectance_mult', 'reflectance_add',
        'radiance_mult', 'radiance_add' per band index, or None if not an MTL dataset.
    """
    raw_header = reader.metadata.raw_header or {}
    mtl_data = raw_header.get("mtl_data")
    if not mtl_data:
        return None

    l1 = mtl_data.get("L1_METADATA_FILE", mtl_data)
    rescaling = l1.get("RADIOMETRIC_RESCALING", {})
    image_attrs = l1.get("IMAGE_ATTRIBUTES", {})

    sun_elev = raw_header.get("sun_elevation") or float(image_attrs.get("SUN_ELEVATION", 0.0) or 0.0)

    # Map discovered bands to their MTL band numbers
    # For Landsat 8/9: B1, B2, B3, B4, B5, B6, B7, B9, B10, B11
    # We inspect band details or band name
    band_details = reader.metadata.band_details or []
    refl_mult = {}
    refl_add = {}
    rad_mult = {}
    rad_add = {}

    for idx, binfo in enumerate(band_details):
        # Map band to its MTL band number. Order matters: more specific tokens must
        # be tested first, because "red" is a substring of "infrared" and would
        # otherwise capture NIR / SWIR / thermal bands.
        b_num = None
        name = binfo.name.lower()
        if "coastal" in name:
            b_num = 1
        elif "blue" in name:
            b_num = 2
        elif "green" in name:
            b_num = 3
        elif "nir" in name or "near infrared" in name:
            b_num = 5
        elif "swir 1" in name or "shortwave infrared 1" in name or "swir1" in name:
            b_num = 6
        elif "swir 2" in name or "shortwave infrared 2" in name or "swir2" in name:
            b_num = 7
        elif "cirrus" in name:
            b_num = 9
        elif "tirs 1" in name or "thermal infrared 1" in name:
            b_num = 10
        elif "tirs 2" in name or "thermal infrared 2" in name:
            b_num = 11
        elif "pan" in name or "panchromatic" in name:
            b_num = 8
        elif "red" in name:
            b_num = 4
        else:
            # Fall back on the band position, but only accept it when the metadata
            # actually carries an MTL band number to read.
            b_num = binfo.mtl_band if getattr(binfo, "mtl_band", None) else idx + 1

        # Look up in rescaling dictionary
        rm = rescaling.get(f"REFLECTANCE_MULT_BAND_{b_num}")
        ra = rescaling.get(f"REFLECTANCE_ADD_BAND_{b_num}")
        lm = rescaling.get(f"RADIANCE_MULT_BAND_{b_num}")
        la = rescaling.get(f"RADIANCE_ADD_BAND_{b_num}")

        refl_mult[idx] = float(rm) if rm is not None else 0.00002
        refl_add[idx] = float(ra) if ra is not None else -0.10000
        rad_mult[idx] = float(lm) if lm is not None else 1.0
        rad_add[idx] = float(la) if la is not None else 0.0

    return {
        "sun_elevation": sun_elev,
        "reflectance_mult": refl_mult,
        "reflectance_add": refl_add,
        "radiance_mult": rad_mult,
        "radiance_add": rad_add,
    }


def calibrate_band_to_reflectance(
    band_data: np.ndarray,
    mult: float,
    add: float,
    sun_elevation_deg: float,
    apply_dos: bool = False,
    dos_percentile: float = 1.0,
    nodata: float = 0.0,
) -> np.ndarray:
    """Convert raw DN band slice to TOA or DOS-corrected surface reflectance.

    Formula:
        rho_prime = mult * DN + add
        rho_toa = rho_prime / sin(sun_elevation)
        rho_surface = max(0, rho_toa - path_radiance)
    """
    valid_mask = (band_data != nodata) & np.isfinite(band_data) & (band_data > 0)
    out = np.zeros_like(band_data, dtype=np.float32)

    if not np.any(valid_mask):
        return out

    # Sun elevation angle in radians
    sun_rad = math.radians(sun_elevation_deg) if sun_elevation_deg > 0 else math.pi / 2.0
    sin_sun = max(0.01, math.sin(sun_rad))

    # Calculate TOA planetary reflectance
    dn_valid = band_data[valid_mask].astype(np.float32)
    rho_toa = (mult * dn_valid + add) / sin_sun

    # Dark Object Subtraction (DOS-1) for atmospheric haze correction
    if apply_dos:
        pos_refl = rho_toa[rho_toa > 0]
        if len(pos_refl) > 0:
            sample_count = min(len(pos_refl), 100000)
            sample_pixels = np.random.choice(pos_refl, size=sample_count, replace=False)
            path_refl = float(np.percentile(sample_pixels, dos_percentile))
            rho_toa = np.maximum(0.0, rho_toa - path_refl)

    out[valid_mask] = rho_toa
    return out


def calibrate_band_to_radiance(
    band_data: np.ndarray,
    mult: float,
    add: float,
    nodata: float = 0.0,
) -> np.ndarray:
    """Convert raw DN band slice to TOA spectral radiance (W / (m^2 * sr * um))."""
    valid_mask = (band_data != nodata) & np.isfinite(band_data) & (band_data > 0)
    out = np.zeros_like(band_data, dtype=np.float32)

    if not np.any(valid_mask):
        return out

    dn_valid = band_data[valid_mask].astype(np.float32)
    out[valid_mask] = mult * dn_valid + add
    return out


def execute_calibration(
    reader: BaseRasterReader,
    cal_type: str = "reflectance",  # "reflectance", "dos", or "radiance"
    band_indices: Optional[List[int]] = None,
    custom_mult: Optional[float] = None,
    custom_add: Optional[float] = None,
    custom_sun_elev: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Run calibration across multiple bands and return a 2D or 3D calibrated float32 array."""
    meta = reader.metadata
    if band_indices is None:
        band_indices = list(range(meta.bands))

    total = len(band_indices)
    cal_params = extract_landsat_cal_params(reader)

    calibrated_bands = []

    for step, b_idx in enumerate(band_indices):
        band_data = reader.read_band(b_idx)

        # Determine parameters. Explicit user-supplied coefficients always take
        # precedence over the MTL-derived defaults, matching how the sun elevation
        # override already behaves.
        if cal_params:
            sun_elev = custom_sun_elev or cal_params["sun_elevation"]
            if cal_type in ("reflectance", "dos"):
                default_mult = cal_params["reflectance_mult"].get(b_idx, 0.00002)
                default_add = cal_params["reflectance_add"].get(b_idx, -0.10000)
            else:
                default_mult = cal_params["radiance_mult"].get(b_idx, 1.0)
                default_add = cal_params["radiance_add"].get(b_idx, 0.0)
        else:
            sun_elev = custom_sun_elev or 45.0
            default_mult = 1.0
            default_add = 0.0

        mult = custom_mult if custom_mult is not None else default_mult
        add = custom_add if custom_add is not None else default_add

        if cal_type == "radiance":
            cal_b = calibrate_band_to_radiance(
                band_data,
                mult=mult,
                add=add,
                nodata=meta.nodata or 0.0,
            )
        else:
            is_dos = (cal_type == "dos")
            cal_b = calibrate_band_to_reflectance(
                band_data,
                mult=mult,
                add=add,
                sun_elevation_deg=sun_elev,
                apply_dos=is_dos,
                nodata=meta.nodata or 0.0,
            )

        calibrated_bands.append(cal_b)
        if progress_callback:
            progress_callback(step + 1, total)

    if len(calibrated_bands) == 1:
        return calibrated_bands[0]
    return np.stack(calibrated_bands, axis=-1)
