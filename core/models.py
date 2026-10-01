"""OpenENVI Core Data Models.

Defines foundational dataclasses and metadata structures for raster layers,
hyperspectral bands, and coordinate referencing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BandInfo:
    """Represents metadata for a single spectral band."""

    index: int
    name: str = ""
    wavelength: Optional[float] = None
    wavelength_unit: str = "nm"
    fwhm: Optional[float] = None
    # Landsat MTL band number (1-11). The same physical band carries different
    # numbers per spacecraft, so the reader records it here rather than leaving
    # downstream code to guess from the band name.
    mtl_band: Optional[int] = None

    def display_name(self) -> str:
        """Return a human-readable display string."""
        if self.wavelength is not None:
            return f"Band {self.index + 1} ({self.wavelength:.2f} {self.wavelength_unit})"
        if self.name:
            return f"Band {self.index + 1}: {self.name}"
        return f"Band {self.index + 1}"


@dataclass
class RasterMetadata:
    """Represents spatial and spectral metadata of a raster dataset."""

    width: int = 0
    height: int = 0
    bands: int = 0
    dtype: str = "float32"
    crs: Optional[str] = None
    transform: Optional[Tuple[float, ...]] = None
    interleave: str = "BSQ"  # 'BSQ', 'BIL', 'BIP'
    nodata: Optional[float] = None
    band_details: List[BandInfo] = field(default_factory=list)
    raw_header: Dict[str, Any] = field(default_factory=dict)
    default_bands: Optional[Tuple[int, ...]] = None

    @property
    def samples(self) -> int:
        """Alias for width (ENVI nomenclature)."""
        return self.width

    @samples.setter
    def samples(self, value: int) -> None:
        self.width = value

    @property
    def lines(self) -> int:
        """Alias for height (ENVI nomenclature)."""
        return self.height

    @lines.setter
    def lines(self, value: int) -> None:
        self.height = value


@dataclass
class RasterLayer:
    """Represents an active or loaded raster layer in OpenENVI."""

    layer_id: str
    name: str
    file_path: str
    metadata: RasterMetadata = field(default_factory=RasterMetadata)
    is_visible: bool = True
    display_mode: str = "grayscale"  # "grayscale" or "rgb"
    active_bands: Tuple[int, ...] = (0,)  # (gray,) or (r, g, b)
    rois: list = field(default_factory=list)

    # Display cache: stores the last stretched uint8 image so that switching
    # back to a layer avoids a full re-read from disk.  Cleared whenever the
    # selected bands or stretch mode changes.  Field excluded from repr to
    # avoid printing potentially large arrays.
    _display_cache: Optional["np.ndarray"] = field(default=None, repr=False)
    _cache_stretch_mode: Optional[str] = field(default=None, repr=False)
    is_thematic: bool = False
    _thematic_image: Optional["np.ndarray"] = field(default=None, repr=False)

    def get_display_cache(self, stretch_mode: str) -> "Optional[np.ndarray]":
        """Return cached display image if it matches the requested stretch mode."""
        if self.is_thematic and self._thematic_image is not None:
            return self._thematic_image
        if self._display_cache is not None and self._cache_stretch_mode == stretch_mode:
            return self._display_cache
        return None

    def set_display_cache(self, image: "np.ndarray", stretch_mode: str) -> None:
        """Store a stretched uint8 display image for later reuse."""
        self._display_cache = image
        self._cache_stretch_mode = stretch_mode

    def set_thematic_image(self, image: "np.ndarray") -> None:
        """Configure layer as a thematic classification layer with fixed RGB visualization."""
        self.is_thematic = True
        self._thematic_image = image
        self._display_cache = image

    def invalidate_display_cache(self) -> None:
        """Discard the cached display image (e.g. after band or stretch change)."""
        if not self.is_thematic:
            self._display_cache = None
            self._cache_stretch_mode = None

