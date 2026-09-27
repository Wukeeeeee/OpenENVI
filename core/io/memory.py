"""OpenENVI In-Memory Raster Reader.

Wraps derived numpy arrays (e.g. Band Math results, Spectral Indices, PCA scores,
classification maps) into the standard BaseRasterReader interface.
"""

from typing import List, Optional, Tuple
import numpy as np

from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterMetadata


class MemoryRasterReader(BaseRasterReader):
    """Raster reader wrapping an in-memory 2D or 3D NumPy array."""

    def __init__(self, data: np.ndarray, name: str = "Memory Layer", parent_metadata: Optional[RasterMetadata] = None):
        super().__init__(file_path=f"memory://{name}")
        self._name = name
        self._parent_meta = parent_metadata

        if data.ndim == 2:
            # (H, W) -> (1, H, W)
            self._cube = data[np.newaxis, :, :].astype(np.float32)
        elif data.ndim == 3:
            if data.shape[-1] <= 10 and data.shape[0] > 10:
                # (H, W, bands) -> (bands, H, W)
                self._cube = np.transpose(data, (2, 0, 1)).astype(np.float32)
            else:
                self._cube = data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported array shape for MemoryRasterReader: {data.shape}")

        self.metadata  # trigger _read_metadata

    def _read_metadata(self) -> RasterMetadata:
        """Create metadata for the in-memory array."""
        bands, lines, samples = self._cube.shape

        band_details = []
        for i in range(bands):
            band_details.append(BandInfo(index=i, name=f"Component {i + 1}"))

        crs = self._parent_meta.crs if self._parent_meta else None
        transform = self._parent_meta.transform if self._parent_meta else None

        meta = RasterMetadata(
            width=samples,
            height=lines,
            bands=bands,
            dtype=str(self._cube.dtype),
            crs=crs,
            transform=transform,
            interleave="BSQ",
            band_details=band_details,
        )
        return meta

    def read_band(self, band_index: int) -> np.ndarray:
        """Read a single 2D band slice."""
        if 0 <= band_index < self._cube.shape[0]:
            return np.array(self._cube[band_index], dtype=np.float32)
        raise IndexError(f"Band index {band_index} out of range [0, {self._cube.shape[0]})")

    def read_pixel_profile(self, x: int, y: int) -> np.ndarray:
        """Read spectral profile across all bands at (x, y)."""
        return np.array(self._cube[:, y, x], dtype=np.float32)

    def pixel_to_geo(self, x: int, y: int) -> Tuple[Optional[float], Optional[float]]:
        """Return parent geospatial coordinates if available."""
        if self._parent_meta and self._parent_meta.transform:
            import rasterio.transform
            geo_x, geo_y = rasterio.transform.xy(self._parent_meta.transform, y, x, offset="center")
            return float(geo_x), float(geo_y)
        return None, None

    def close(self) -> None:
        """No persistent file handles to close."""
        pass
