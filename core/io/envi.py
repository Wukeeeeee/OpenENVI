"""OpenENVI ENVI Standard Format I/O Engine.

Provides high-performance, memory-mapped reading of ENVI standard hyperspectral
and multispectral raster cubes (.hdr, .dat, .raw, .bsq, .bil, .bip).
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterMetadata


ENVI_DATA_TYPES = {
    1: np.dtype("uint8"),
    2: np.dtype("int16"),
    3: np.dtype("int32"),
    4: np.dtype("float32"),
    5: np.dtype("float64"),
    6: np.dtype("complex64"),
    9: np.dtype("complex128"),
    12: np.dtype("uint16"),
    13: np.dtype("uint32"),
    14: np.dtype("int64"),
    15: np.dtype("uint64"),
}


def parse_envi_header(header_path: str) -> Dict[str, Any]:
    """Parse an ENVI .hdr file into a dictionary of key-value pairs.

    Args:
        header_path: Path to the .hdr file.

    Returns:
        Dictionary containing parsed ENVI header keys.
    """
    with open(header_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    header_dict: Dict[str, Any] = {}
    # Remove leading 'ENVI' header tag
    content = re.sub(r"^\s*ENVI\b", "", content, flags=re.IGNORECASE).strip()

    # Match key = value, taking into account multi-line values enclosed in {}
    pattern = re.compile(r"([a-zA-Z0-9_\s]+?)\s*=\s*(\{([^}]*)\}|[^\r\n]*)", re.MULTILINE)
    for match in pattern.finditer(content):
        key = match.group(1).strip().lower()
        if match.group(3) is not None:
            # Multi-line or comma-separated list in braces
            raw_val = match.group(3).strip()
            # Split items by comma or newline
            items = [item.strip() for item in re.split(r"[,\n]+", raw_val) if item.strip()]
            header_dict[key] = items
        else:
            val = match.group(2).strip()
            header_dict[key] = val

    return header_dict


class ENVIRasterReader(BaseRasterReader):
    """Memory-efficient raster reader for ENVI format images using numpy memmap."""

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.header_path, self.data_path = self._resolve_paths(file_path)
        self._raw_header: Dict[str, Any] = {}
        self._memmap: Optional[np.ndarray] = None
        self._interleave = "bsq"
        self._lines = 0
        self._samples = 0
        self._bands = 0
        self._dtype = np.dtype("float32")
        self._map_info: Optional[Dict[str, Any]] = None

        # Load metadata and initialize memmap
        self.metadata  # trigger _read_metadata

    def _resolve_paths(self, path: str) -> Tuple[str, str]:
        """Resolve corresponding .hdr header file and binary data file paths."""
        base, ext = os.path.splitext(path)
        ext_lower = ext.lower()

        if ext_lower == ".hdr":
            header_path = path
            # Search for candidate binary files
            for candidate_ext in [
                ".dat", ".DAT", ".raw", ".RAW", ".bsq", ".BSQ",
                ".bil", ".BIL", ".bip", ".BIP", ".img", ".IMG", ""
            ]:
                candidate = base + candidate_ext
                if os.path.isfile(candidate) and candidate != header_path:
                    return header_path, candidate
            # If no binary file found with known ext, default to base (extensionless)
            return header_path, base
        else:
            data_path = path
            header_path = base + ".hdr"
            if not os.path.isfile(header_path):
                # Check if path.hdr exists (e.g. image.dat.hdr)
                alt_hdr = path + ".hdr"
                if os.path.isfile(alt_hdr):
                    header_path = alt_hdr
            return header_path, data_path

    def _read_metadata(self) -> RasterMetadata:
        """Parse the ENVI header and initialize the memory map."""
        if not os.path.isfile(self.header_path):
            raise FileNotFoundError(f"ENVI header file not found: {self.header_path}")

        self._raw_header = parse_envi_header(self.header_path)

        self._samples = int(self._raw_header.get("samples", 0))
        self._lines = int(self._raw_header.get("lines", 0))
        self._bands = int(self._raw_header.get("bands", 1))
        self._interleave = str(self._raw_header.get("interleave", "bsq")).lower()

        dtype_code = int(self._raw_header.get("data type", 4))
        base_dtype = ENVI_DATA_TYPES.get(dtype_code, np.dtype("float32"))

        byte_order = int(self._raw_header.get("byte order", 0))
        endianness = "<" if byte_order == 0 else ">"
        self._dtype = base_dtype.newbyteorder(endianness)

        offset = int(self._raw_header.get("header offset", 0))

        # Parse Map Info if available
        self._parse_map_info()

        # Parse Wavelengths and Band Names
        band_details = self._parse_band_details()

        # Open memory-mapped array if binary file exists
        if os.path.isfile(self.data_path):
            if self._interleave == "bsq":
                shape = (self._bands, self._lines, self._samples)
            elif self._interleave == "bil":
                shape = (self._lines, self._bands, self._samples)
            elif self._interleave == "bip":
                shape = (self._lines, self._samples, self._bands)
            else:
                shape = (self._bands, self._lines, self._samples)

            self._memmap = np.memmap(
                self.data_path,
                dtype=self._dtype,
                mode="r",
                offset=offset,
                shape=shape,
            )

        meta = RasterMetadata(
            width=self._samples,
            height=self._lines,
            bands=self._bands,
            dtype=str(self._dtype),
            crs=self._map_info.get("projection") if self._map_info else None,
            interleave=self._interleave.upper(),
            band_details=band_details,
            raw_header=self._raw_header,
        )
        return meta

    def _parse_band_details(self) -> List[BandInfo]:
        """Extract wavelength and band name information from header."""
        band_details = []
        raw_wavelengths = self._raw_header.get("wavelength", [])
        raw_names = self._raw_header.get("band names", [])
        raw_fwhm = self._raw_header.get("fwhm", [])
        unit = str(self._raw_header.get("wavelength units", "nm"))

        for i in range(self._bands):
            wl = None
            if i < len(raw_wavelengths):
                try:
                    wl = float(raw_wavelengths[i])
                except ValueError:
                    pass

            name = ""
            if i < len(raw_names):
                name = str(raw_names[i])

            fwhm = None
            if i < len(raw_fwhm):
                try:
                    fwhm = float(raw_fwhm[i])
                except ValueError:
                    pass

            band_details.append(
                BandInfo(
                    index=i,
                    name=name,
                    wavelength=wl,
                    wavelength_unit=unit,
                    fwhm=fwhm,
                )
            )
        return band_details

    def _parse_map_info(self) -> None:
        """Parse ENVI 'map info' field for geospatial referencing."""
        raw = self._raw_header.get("map info")
        if not raw or not isinstance(raw, list) or len(raw) < 7:
            return

        try:
            proj = raw[0].strip()
            tie_x = float(raw[1])
            tie_y = float(raw[2])
            easting = float(raw[3])
            northing = float(raw[4])
            dx = float(raw[5])
            dy = float(raw[6])

            self._map_info = {
                "projection": proj,
                "tie_x": tie_x,
                "tie_y": tie_y,
                "easting": easting,
                "northing": northing,
                "dx": dx,
                "dy": dy,
            }
        except (ValueError, IndexError):
            self._map_info = None

    def read_band(self, band_index: int) -> np.ndarray:
        """Read a single 2D band slice (height, width)."""
        if self._memmap is None:
            raise RuntimeError(f"Data file not loaded for {self.file_path}")

        if band_index < 0 or band_index >= self._bands:
            raise IndexError(f"Band index {band_index} out of range [0, {self._bands})")

        if self._interleave == "bsq":
            slice_data = self._memmap[band_index, :, :]
        elif self._interleave == "bil":
            slice_data = self._memmap[:, band_index, :]
        elif self._interleave == "bip":
            slice_data = self._memmap[:, :, band_index]
        else:
            slice_data = self._memmap[band_index, :, :]

        return np.array(slice_data, dtype=np.float32)

    def read_pixel_profile(self, x: int, y: int) -> np.ndarray:
        """Read spectral profile across all bands at (x, y)."""
        if self._memmap is None:
            raise RuntimeError(f"Data file not loaded for {self.file_path}")

        if not (0 <= x < self._samples and 0 <= y < self._lines):
            raise IndexError(f"Coordinates ({x}, {y}) out of range")

        if self._interleave == "bsq":
            curve = self._memmap[:, y, x]
        elif self._interleave == "bil":
            curve = self._memmap[y, :, x]
        elif self._interleave == "bip":
            curve = self._memmap[y, x, :]
        else:
            curve = self._memmap[:, y, x]

        return np.array(curve, dtype=np.float32)

    def pixel_to_geo(self, x: int, y: int) -> Tuple[Optional[float], Optional[float]]:
        """Compute geographic coordinates from pixel coordinates."""
        if not self._map_info:
            return None, None

        tie_x = self._map_info["tie_x"]
        tie_y = self._map_info["tie_y"]
        easting = self._map_info["easting"]
        northing = self._map_info["northing"]
        dx = self._map_info["dx"]
        dy = self._map_info["dy"]

        # Note: ENVI 1-based pixel tie-points
        geo_x = easting + (x - (tie_x - 1.0)) * dx
        geo_y = northing - (y - (tie_y - 1.0)) * dy
        return float(geo_x), float(geo_y)

    def close(self) -> None:
        """Close memory-mapped file."""
        if self._memmap is not None:
            # Delete reference to release Windows file lock
            del self._memmap
            self._memmap = None
