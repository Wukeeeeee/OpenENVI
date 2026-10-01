"""OpenENVI Landsat Satellite Product Package Reader.

Parses USGS Landsat metadata files (*_MTL.txt) to automatically package multi-band
satellite observations into a coherent raster dataset with calibrated wavelengths,
band nomenclature, natural color RGB assignments, and lazy windowed I/O.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import core.proj_setup  # Ensure PROJ library paths are configured before rasterio
import rasterio
from rasterio.windows import Window

from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterMetadata


# Standard Spectral Configurations for Landsat Sensors
LANDSAT_8_9_BANDS = [
    ("FILE_NAME_BAND_1", "Coastal Aerosol", 443.0),
    ("FILE_NAME_BAND_2", "Blue", 482.0),
    ("FILE_NAME_BAND_3", "Green", 561.0),
    ("FILE_NAME_BAND_4", "Red", 655.0),
    ("FILE_NAME_BAND_5", "Near Infrared (NIR)", 865.0),
    ("FILE_NAME_BAND_6", "Shortwave Infrared 1 (SWIR 1)", 1609.0),
    ("FILE_NAME_BAND_7", "Shortwave Infrared 2 (SWIR 2)", 2201.0),
    ("FILE_NAME_BAND_9", "Cirrus", 1373.0),
    ("FILE_NAME_BAND_10", "Thermal Infrared 1 (TIRS 1)", 10895.0),
    ("FILE_NAME_BAND_11", "Thermal Infrared 2 (TIRS 2)", 12005.0),
]

LANDSAT_4_5_7_BANDS = [
    ("FILE_NAME_BAND_1", "Blue", 485.0),
    ("FILE_NAME_BAND_2", "Green", 560.0),
    ("FILE_NAME_BAND_3", "Red", 660.0),
    ("FILE_NAME_BAND_4", "Near Infrared (NIR)", 830.0),
    ("FILE_NAME_BAND_5", "Shortwave Infrared 1 (SWIR 1)", 1650.0),
    ("FILE_NAME_BAND_7", "Shortwave Infrared 2 (SWIR 2)", 2215.0),
    ("FILE_NAME_BAND_6", "Thermal Infrared", 11450.0),
]


def parse_mtl_file(file_path: str) -> Dict[str, Any]:
    """Parse hierarchical USGS MTL text file into a nested Python dictionary."""
    root: Dict[str, Any] = {}
    stack: List[Dict[str, Any]] = [root]

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line == "END":
                continue
            if line.startswith("GROUP = "):
                group_name = line.split("=", 1)[1].strip()
                new_group: Dict[str, Any] = {}
                stack[-1][group_name] = new_group
                stack.append(new_group)
            elif line.startswith("END_GROUP = "):
                if len(stack) > 1:
                    stack.pop()
            elif "=" in line:
                key, val = [x.strip() for x in line.split("=", 1)]
                val = val.strip('"')
                stack[-1][key] = val

    return root


class LandsatMTLReader(BaseRasterReader):
    """Raster reader for Landsat product metadata packages (*_MTL.txt)."""

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.directory = os.path.dirname(os.path.abspath(file_path))
        self.mtl_data = parse_mtl_file(file_path)

        # Open rasterio dataset cache: band_idx -> DatasetReader
        self._datasets: Dict[int, rasterio.io.DatasetReader] = {}
        self._band_paths: List[str] = []
        self._ref_dataset: Optional[rasterio.io.DatasetReader] = None

        self.metadata  # Trigger _read_metadata

    def _read_metadata(self) -> RasterMetadata:
        """Parse MTL structures, discover band GeoTIFFs, and build RasterMetadata."""
        l1 = self.mtl_data.get("L1_METADATA_FILE", self.mtl_data)
        prod = l1.get("PRODUCT_METADATA", {})

        spacecraft = str(prod.get("SPACECRAFT_ID", "")).upper()
        is_l8_9 = "LANDSAT_8" in spacecraft or "LANDSAT_9" in spacecraft or "LANDSAT 8" in spacecraft

        if is_l8_9:
            band_templates = LANDSAT_8_9_BANDS
            default_rgb = (3, 2, 1)  # Red (B4), Green (B3), Blue (B2)
        else:
            band_templates = LANDSAT_4_5_7_BANDS
            default_rgb = (2, 1, 0)  # Red (B3), Green (B2), Blue (B1)

        discovered_bands: List[BandInfo] = []
        self._band_paths = []

        for mtl_key, default_name, wavelength in band_templates:
            filename = prod.get(mtl_key)
            if not filename:
                continue

            full_path = os.path.join(self.directory, filename)
            if not os.path.isfile(full_path):
                # Try case-insensitive fallback in directory
                cand = None
                for f in os.listdir(self.directory):
                    if f.lower() == filename.lower():
                        cand = os.path.join(self.directory, f)
                        break
                if cand and os.path.isfile(cand):
                    full_path = cand
                else:
                    continue

            band_idx = len(self._band_paths)
            self._band_paths.append(full_path)
            # The numeric suffix of FILE_NAME_BAND_n is the MTL band number, which
            # is exactly what REFLECTANCE_MULT_BAND_n in the metadata is keyed on.
            mtl_band = int(mtl_key.rsplit("_", 1)[-1])
            discovered_bands.append(
                BandInfo(
                    index=band_idx,
                    name=default_name,
                    wavelength=wavelength,
                    wavelength_unit="nm",
                    mtl_band=mtl_band,
                )
            )

        if not self._band_paths:
            raise ValueError(f"No corresponding Landsat band GeoTIFF files found in '{self.directory}'.")

        # Open reference dataset (Band 0) to extract spatial referencing
        self._ref_dataset = rasterio.open(self._band_paths[0])
        self._datasets[0] = self._ref_dataset

        crs_str = self._ref_dataset.crs.to_string() if self._ref_dataset.crs else None
        transform_obj = self._ref_dataset.transform if self._ref_dataset.transform else None

        # Build raw header dictionary containing calibration attributes.
        # A truncated or locale-formatted MTL makes float() raise on
        # SUN_ELEVATION, and this runs after the GDAL handle is owned above --
        # close() is never reached on that path, so the file stayed locked.
        try:
            sun_elevation = float(l1.get("IMAGE_ATTRIBUTES", {}).get("SUN_ELEVATION", 0.0) or 0.0)
            earth_sun_distance = float(l1.get("IMAGE_ATTRIBUTES", {}).get("EARTH_SUN_DISTANCE", 1.0) or 1.0)
        except (TypeError, ValueError):
            sun_elevation, earth_sun_distance = 0.0, 1.0

        raw_header = {
            "spacecraft": spacecraft,
            "sensor": prod.get("SENSOR_ID", ""),
            "date_acquired": prod.get("DATE_ACQUIRED", ""),
            "scene_center_time": prod.get("SCENE_CENTER_TIME", ""),
            "sun_elevation": sun_elevation,
            "earth_sun_distance": earth_sun_distance,
            "mtl_data": self.mtl_data,
        }

        # Validate default_rgb indices
        valid_default_rgb = default_rgb if (len(self._band_paths) > max(default_rgb)) else None

        meta = RasterMetadata(
            width=self._ref_dataset.width,
            height=self._ref_dataset.height,
            bands=len(self._band_paths),
            dtype="float32",
            crs=crs_str,
            transform=transform_obj,
            interleave="BSQ",
            nodata=float(self._ref_dataset.nodata) if self._ref_dataset.nodata is not None else 0.0,
            band_details=discovered_bands,
            raw_header=raw_header,
            default_bands=valid_default_rgb,
        )
        return meta

    def _get_dataset(self, band_index: int) -> rasterio.io.DatasetReader:
        """Get or lazily open the rasterio DatasetReader for the given band index."""
        if band_index < 0 or band_index >= len(self._band_paths):
            raise IndexError(f"Band index {band_index} out of range [0, {len(self._band_paths)})")

        if band_index not in self._datasets:
            self._datasets[band_index] = rasterio.open(self._band_paths[band_index])
        return self._datasets[band_index]

    def read_band(self, band_index: int) -> np.ndarray:
        """Read a single 2D band slice (height, width) as float32.

        Uses rasterio's ``out=`` parameter to decode directly into a pre-allocated
        float32 buffer, eliminating the intermediate uint16/uint8 copy (saves
        ~110 MB per band on a full Landsat scene).
        """
        ds = self._get_dataset(band_index)
        buf = np.empty((ds.height, ds.width), dtype=np.float32)
        ds.read(1, out=buf)  # rasterio converts source dtype → float32 in-place
        return buf

    def read_pixel_profile(self, x: int, y: int) -> np.ndarray:
        """Read spectral profile across all packaged bands at (x, y) using 1x1 window."""
        if not (0 <= x < self.metadata.width and 0 <= y < self.metadata.height):
            raise IndexError(f"Coordinates ({x}, {y}) out of range (0..{self.metadata.width}, 0..{self.metadata.height})")

        window = Window(x, y, 1, 1)
        profile_vals = []
        for i in range(len(self._band_paths)):
            ds = self._get_dataset(i)
            val = ds.read(1, window=window)[0, 0]
            profile_vals.append(val)

        return np.array(profile_vals, dtype=np.float32)

    def pixel_to_geo(self, x: int, y: int) -> Tuple[Optional[float], Optional[float]]:
        """Transform pixel coordinate to geospatial coordinate."""
        if not self._ref_dataset or not self._ref_dataset.transform:
            return None, None
        try:
            geo_x, geo_y = rasterio.transform.xy(self._ref_dataset.transform, y, x, offset="center")
            return float(geo_x), float(geo_y)
        except Exception:
            return None, None

    def close(self) -> None:
        """Close all open rasterio dataset handles."""
        for ds in self._datasets.values():
            try:
                if ds and not ds.closed:
                    ds.close()
            except Exception:
                pass
        self._datasets.clear()
        self._ref_dataset = None
