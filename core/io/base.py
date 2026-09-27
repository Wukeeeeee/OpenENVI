"""OpenENVI Raster Reader Base Specification.

Defines the abstract interface for all raster file formats supported by OpenENVI,
ensuring lazy loading, memory efficiency, and coordinate transformation capabilities.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np

from core.models import RasterMetadata


class BaseRasterReader(ABC):
    """Abstract base class for remote sensing raster data readers."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._metadata: Optional[RasterMetadata] = None

    @property
    def metadata(self) -> RasterMetadata:
        """Return the raster metadata structure."""
        if self._metadata is None:
            self._metadata = self._read_metadata()
        return self._metadata

    @abstractmethod
    def _read_metadata(self) -> RasterMetadata:
        """Parse file headers and populate RasterMetadata."""
        pass

    @abstractmethod
    def read_band(self, band_index: int) -> np.ndarray:
        """Read a single 2D band slice.

        Args:
            band_index: 0-based band index.

        Returns:
            2D numpy array of shape (height, width).
        """
        pass

    def read_bands(self, band_indices: List[int]) -> np.ndarray:
        """Read multiple bands into a 3D array (height, width, len(band_indices)).

        Args:
            band_indices: List of 0-based band indices.

        Returns:
            3D numpy array of shape (height, width, len(band_indices)).
        """
        bands = [self.read_band(idx) for idx in band_indices]
        return np.stack(bands, axis=-1)

    @abstractmethod
    def read_pixel_profile(self, x: int, y: int) -> np.ndarray:
        """Read the spectral profile across all bands at pixel coordinates (x, y).

        Args:
            x: 0-based pixel column (sample).
            y: 0-based pixel row (line).

        Returns:
            1D numpy array of shape (bands,).
        """
        pass

    @abstractmethod
    def pixel_to_geo(self, x: int, y: int) -> Tuple[Optional[float], Optional[float]]:
        """Transform pixel coordinate (x, y) to geospatial coordinate (easting/lon, northing/lat).

        Args:
            x: Pixel X (sample, 0-based).
            y: Pixel Y (line, 0-based).

        Returns:
            Tuple of (geo_x, geo_y) or (None, None) if not georeferenced.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release open file handles and memory resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
