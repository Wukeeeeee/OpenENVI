"""OpenENVI Region of Interest (ROI) Module.

Provides ROI definition, polygon/rectangle raster mask generation,
statistical calculation (min, max, mean, std), and mean spectral extraction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ROI:
    """Represents a Region of Interest with polygon/box coordinates and color."""

    roi_id: str
    name: str
    color: str = "#ff0000"  # Hex color string
    # List of (x, y) polygon vertices
    polygon_points: List[Tuple[float, float]] = field(default_factory=list)
    # Bounding box (x_min, y_min, x_max, y_max)
    bbox: Optional[Tuple[int, int, int, int]] = None

    def get_mask(self, height: int, width: int) -> np.ndarray:
        """Create a 2D boolean mask of shape (height, width) for this ROI."""
        mask = np.zeros((height, width), dtype=bool)

        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            x0 = max(0, min(width - 1, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height - 1, y0))
            y1 = max(0, min(height, y1))
            mask[y0:y1, x0:x1] = True

        elif len(self.polygon_points) >= 3:
            from matplotlib.path import Path
            poly = Path(self.polygon_points)
            y, x = np.mgrid[:height, :width]
            points = np.vstack((x.flatten(), y.flatten())).T
            mask = poly.contains_points(points).reshape((height, width))

        return mask

    def calculate_statistics(self, raster_band: np.ndarray) -> Dict[str, float]:
        """Calculate statistical metrics for the ROI within a 2D band slice.

        Args:
            raster_band: 2D numpy array of shape (height, width).

        Returns:
            Dictionary with count, min, max, mean, and std.
        """
        mask = self.get_mask(raster_band.shape[0], raster_band.shape[1])
        pixels = raster_band[mask]
        valid_pixels = pixels[np.isfinite(pixels)]

        if len(valid_pixels) == 0:
            return {
                "count": 0,
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
            }

        return {
            "count": int(len(valid_pixels)),
            "min": float(np.min(valid_pixels)),
            "max": float(np.max(valid_pixels)),
            "mean": float(np.mean(valid_pixels)),
            "std": float(np.std(valid_pixels)),
        }

    def calculate_mean_spectrum(
        self,
        reader,
    ) -> Optional[np.ndarray]:
        """Calculate mean spectrum across all bands for pixels inside this ROI.

        Args:
            reader: BaseRasterReader instance.

        Returns:
            1D numpy array of shape (bands,) representing mean reflectance/DN.
        """
        meta = reader.metadata
        mask = self.get_mask(meta.height, meta.width)
        num_pixels = np.sum(mask)

        if num_pixels == 0:
            return None

        # Sample or accumulate across bands
        mean_spectrum = np.zeros(meta.bands, dtype=np.float32)
        for b in range(meta.bands):
            band_data = reader.read_band(b)
            valid = band_data[mask]
            mean_spectrum[b] = float(np.mean(valid[np.isfinite(valid)]))

        return mean_spectrum
