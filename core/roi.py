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
    # List of polygons, where each polygon is a list of (x, y) vertex tuples
    polygons: List[List[Tuple[float, float]]] = field(default_factory=list)
    # Bounding box (x_min, y_min, x_max, y_max)
    bbox: Optional[Tuple[int, int, int, int]] = None
    # Backwards compatibility: initial single polygon points
    polygon_points: List[Tuple[float, float]] = field(default_factory=list)
    # Visibility state on map
    is_visible: bool = True

    def __post_init__(self):
        # If legacy polygon_points was passed, ensure it is added into polygons
        if self.polygon_points and not self.polygons:
            if (
                isinstance(self.polygon_points, list)
                and len(self.polygon_points) > 0
                and isinstance(self.polygon_points[0], (list, tuple))
                and len(self.polygon_points[0]) > 0
                and isinstance(self.polygon_points[0][0], (list, tuple))
            ):
                self.polygons = [[(float(pt[0]), float(pt[1])) for pt in poly] for poly in self.polygon_points]
                self.polygon_points = list(self.polygons[0])
            else:
                pts = [(float(pt[0]), float(pt[1])) for pt in self.polygon_points]
                self.polygons.append(pts)
                self.polygon_points = pts
        elif self.polygons and not self.polygon_points:
            self.polygon_points = list(self.polygons[0])

    def add_polygon(self, points: List[Tuple[float, float]]) -> None:
        """Add a closed polygon vertex list to this ROI."""
        if len(points) >= 3:
            self.polygons.append(list(points))
            self.polygon_points = list(points)

    def remove_last_polygon(self) -> bool:
        """Remove the most recently added polygon."""
        if self.polygons:
            self.polygons.pop()
            self.polygon_points = list(self.polygons[0]) if self.polygons else []
            return True
        return False

    def clear_polygons(self) -> None:
        """Clear all polygons in this ROI."""
        self.polygons.clear()
        self.polygon_points.clear()

    def to_dict(self) -> dict:
        """Serialize ROI to dictionary."""
        return {
            "roi_id": self.roi_id,
            "name": self.name,
            "color": self.color,
            "polygons": [
                [list(p) for p in poly] for poly in self.polygons
            ],
            "bbox": list(self.bbox) if self.bbox else None,
            "is_visible": self.is_visible,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ROI":
        """Deserialize ROI from dictionary."""
        polys = [
            [(float(p[0]), float(p[1])) for p in poly]
            for poly in d.get("polygons", [])
        ]
        bbox = tuple(d["bbox"]) if d.get("bbox") else None
        return cls(
            roi_id=d.get("roi_id", "roi_custom"),
            name=d.get("name", "ROI"),
            color=d.get("color", "#2ecc71"),
            polygons=polys,
            bbox=bbox,
            is_visible=d.get("is_visible", True),
        )


    @property
    def num_polygons(self) -> int:
        """Total number of polygons in this ROI class."""
        count = len(self.polygons)
        if count == 0 and self.bbox is not None:
            return 1
        return count

    def get_mask(self, height: int, width: int) -> np.ndarray:
        """Create a 2D boolean mask of shape (height, width) for all polygons/boxes in this ROI."""
        mask = np.zeros((height, width), dtype=bool)

        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            x0 = max(0, min(width - 1, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height - 1, y0))
            y1 = max(0, min(height, y1))
            mask[y0:y1, x0:x1] = True

        for pts_list in self.polygons:
            if len(pts_list) >= 3:
                pts = np.asarray(pts_list)
                min_x = max(0, int(np.floor(np.min(pts[:, 0]))))
                max_x = min(width, int(np.ceil(np.max(pts[:, 0]))))
                min_y = max(0, int(np.floor(np.min(pts[:, 1]))))
                max_y = min(height, int(np.ceil(np.max(pts[:, 1]))))

                if max_x > min_x and max_y > min_y:
                    from matplotlib.path import Path
                    poly = Path(pts)
                    box_y, box_x = np.mgrid[min_y:max_y, min_x:max_x]
                    box_points = np.column_stack((box_x.ravel(), box_y.ravel()))
                    sub_mask = poly.contains_points(box_points).reshape((max_y - min_y, max_x - min_x))
                    mask[min_y:max_y, min_x:max_x] |= sub_mask

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
            valid_finite = valid[np.isfinite(valid)]
            mean_spectrum[b] = float(np.mean(valid_finite)) if len(valid_finite) > 0 else 0.0

        return mean_spectrum
