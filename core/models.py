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
