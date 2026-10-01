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

    def __init__(
        self,
        data: np.ndarray,
        name: str = "Memory Layer",
        parent_metadata: Optional[RasterMetadata] = None,
        layout: Optional[str] = None,
    ):
        """Wrap an in-memory raster.

        Args:
            data: 2D (lines, samples) or 3D array.
            name: Layer name.
            parent_metadata: Metadata of the source layer, used to resolve the
                layout of a 3D array and to inherit CRS/transform/band names.
            layout: One of ``"bhw"`` for (bands, lines, samples), ``"hwb"`` for
                (lines, samples, bands), or ``None`` to infer it.
        """
        super().__init__(file_path=f"memory://{name}")
        self._name = name
        self._parent_meta = parent_metadata

        if layout not in (None, "bhw", "hwb"):
            raise ValueError(f"Unsupported layout {layout!r}; expected 'bhw', 'hwb' or None.")

        if data.ndim == 2:
            # (H, W) -> (1, H, W)
            self._cube = data[np.newaxis, :, :].astype(np.float32)
        elif data.ndim == 3:
            if layout == "hwb":
                self._cube = np.transpose(data, (2, 0, 1)).astype(np.float32)
            elif layout == "bhw":
                self._cube = data.astype(np.float32)
            elif self._parent_meta and data.shape[0] == self._parent_meta.height and data.shape[1] == self._parent_meta.width:
                self._cube = np.transpose(data, (2, 0, 1)).astype(np.float32)
            elif self._parent_meta and data.shape[1] == self._parent_meta.height and data.shape[2] == self._parent_meta.width:
                self._cube = data.astype(np.float32)
            elif data.shape[-1] < min(data.shape[0], data.shape[1]):
                self._cube = np.transpose(data, (2, 0, 1)).astype(np.float32)
            elif data.shape[0] < min(data.shape[1], data.shape[2]):
                self._cube = data.astype(np.float32)
            else:
                # Neither other axis is the smallest, so this cannot be read as a
                # small-band (H, W, bands) image either -- treat it as
                # (bands, H, W), which is what core/algorithms emits. The old
                # "last axis <= 100 means bands" fallback used to sit here and
                # inverted every small cube: an 8-band 4x5 chip was read as 5
                # bands over an 8x4 image, silently changing both the band count
                # and the geometry. A 3D array that really is (H, W, bands) with
                # more bands than lines or samples is still ambiguous here; pass
                # layout= explicitly, or parent_metadata, to settle it.
                self._cube = data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported array shape for MemoryRasterReader: {data.shape}")

        self.metadata  # trigger _read_metadata

    def _read_metadata(self) -> RasterMetadata:
        """Create metadata for the in-memory array."""
        bands, lines, samples = self._cube.shape

        if self._parent_meta and self._parent_meta.band_details and len(self._parent_meta.band_details) == bands:
            band_details = list(self._parent_meta.band_details)
        else:
            band_details = []
            for i in range(bands):
                b_name = f"Band {i + 1}"
                wl = None
                wl_unit = "nm"
                fwhm = None
                if self._parent_meta and self._parent_meta.band_details and i < len(self._parent_meta.band_details):
                    p_info = self._parent_meta.band_details[i]
                    b_name = p_info.name or b_name
                    wl = p_info.wavelength
                    wl_unit = p_info.wavelength_unit
                    fwhm = p_info.fwhm
                band_details.append(BandInfo(index=i, name=b_name, wavelength=wl, wavelength_unit=wl_unit, fwhm=fwhm))

        crs = self._parent_meta.crs if self._parent_meta else None
        transform = self._parent_meta.transform if self._parent_meta else None
        nodata = self._parent_meta.nodata if self._parent_meta else None
        raw_header = dict(self._parent_meta.raw_header) if self._parent_meta and self._parent_meta.raw_header else {}
        default_bands = self._parent_meta.default_bands if self._parent_meta else None

        meta = RasterMetadata(
            width=samples,
            height=lines,
            bands=bands,
            dtype=str(self._cube.dtype),
            crs=crs,
            transform=transform,
            interleave="BSQ",
            nodata=nodata,
            band_details=band_details,
            raw_header=raw_header,
            default_bands=default_bands,
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
            try:
                import rasterio.transform
                from affine import Affine
                t = self._parent_meta.transform
                if isinstance(t, (tuple, list)):
                    if len(t) == 6:
                        t = Affine(*t)
                    elif len(t) == 9:
                        t = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
                geo_x, geo_y = rasterio.transform.xy(t, y, x, offset="center")
                return float(geo_x), float(geo_y)
            except Exception:
                return None, None
        return None, None

    def close(self) -> None:
        """No persistent file handles to close."""
        pass
