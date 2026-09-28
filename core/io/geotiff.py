"""OpenENVI GeoTIFF & Multi-format Raster I/O Engine.

Wraps Rasterio for high-performance reading of GeoTIFF, BigTIFF, and other GDAL formats.
"""

from typing import List, Optional, Tuple
import numpy as np

import core.proj_setup  # Configure PROJ paths before importing rasterio
import rasterio
from rasterio.windows import Window

from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterMetadata


class GeoTIFFRasterReader(BaseRasterReader):
    """Raster reader for GeoTIFF and GDAL formats using Rasterio."""

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._dataset = rasterio.open(file_path)
        self.metadata  # trigger _read_metadata

    def _read_metadata(self) -> RasterMetadata:
        """Extract metadata from the open rasterio dataset."""
        ds = self._dataset

        band_details = []
        for i in range(ds.count):
            desc = ds.descriptions[i] if ds.descriptions and i < len(ds.descriptions) else ""
            wl = None
            try:
                tags = ds.tags(i + 1)
                wl_str = tags.get("WAVELENGTH") or tags.get("wavelength")
                if wl_str:
                    wl = float(wl_str)
            except Exception:
                pass

            band_details.append(
                BandInfo(
                    index=i,
                    name=desc or f"Band {i + 1}",
                    wavelength=wl,
                )
            )

        crs_str = ds.crs.to_string() if ds.crs else None
        transform_tuple = tuple(ds.transform) if ds.transform else None

        interleave_val = getattr(ds, "interleaving", None)
        interleave_mode = "BIP" if (interleave_val and str(interleave_val).lower().endswith("pixel")) else "BSQ"

        meta = RasterMetadata(
            width=ds.width,
            height=ds.height,
            bands=ds.count,
            dtype=ds.dtypes[0] if ds.dtypes else "float32",
            crs=crs_str,
            transform=transform_tuple,
            interleave=interleave_mode,
            nodata=float(ds.nodata) if ds.nodata is not None else None,
            band_details=band_details,
        )
        return meta

    def read_band(self, band_index: int) -> np.ndarray:
        """Read a single 2D band slice (height, width).

        Args:
            band_index: 0-based band index.
        """
        if band_index < 0 or band_index >= self._dataset.count:
            raise IndexError(f"Band index {band_index} out of range [0, {self._dataset.count})")

        # Rasterio uses 1-based indexing for bands
        band_data = self._dataset.read(band_index + 1)
        return band_data.astype(np.float32)

    def read_pixel_profile(self, x: int, y: int) -> np.ndarray:
        """Read spectral profile across all bands at (x, y)."""
        if not (0 <= x < self._dataset.width and 0 <= y < self._dataset.height):
            raise IndexError(f"Coordinates ({x}, {y}) out of range")

        # Read 1x1 window across all bands
        window = Window(x, y, 1, 1)
        pixel_cube = self._dataset.read(window=window)  # shape (bands, 1, 1)
        return pixel_cube[:, 0, 0].astype(np.float32)

    def pixel_to_geo(self, x: int, y: int) -> Tuple[Optional[float], Optional[float]]:
        """Compute geographic coordinates from pixel coordinates."""
        if not self._dataset.transform:
            return None, None

        try:
            # rasterio.transform.xy takes (row, col)
            geo_x, geo_y = rasterio.transform.xy(self._dataset.transform, y, x, offset="center")
            return float(geo_x), float(geo_y)
        except Exception:
            return None, None

    def close(self) -> None:
        """Close open rasterio dataset."""
        if self._dataset is not None and not self._dataset.closed:
            self._dataset.close()
