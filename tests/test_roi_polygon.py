"""Unit tests for ROI polygon drawing, mask extraction, and spectral calculation."""

import pytest
import numpy as np

from core.roi import ROI
from core.io.memory import MemoryRasterReader


def test_polygon_roi_mask():
    """Test generating a 2D boolean mask from polygon coordinates."""
    # Triangular polygon: (2, 2), (8, 2), (5, 8)
    poly_pts = [(2.0, 2.0), (8.0, 2.0), (5.0, 8.0)]
    roi = ROI(roi_id="poly_1", name="Triangle ROI", color="#ff0000", polygon_points=poly_pts)

    mask = roi.get_mask(height=10, width=10)
    assert mask.shape == (10, 10)
    assert mask.dtype == bool
    # Center of triangle (5, 4) should be True
    assert bool(mask[4, 5]) or bool(mask[3, 5])
    # Outside corner (0, 0) should be False
    assert not bool(mask[0, 0])
    assert not bool(mask[9, 9])


def test_polygon_roi_statistics_and_spectrum():
    """Test multi-band statistics and mean spectrum calculation from polygon ROI."""
    b1 = np.full((10, 10), 10.0, dtype=np.float32)
    b2 = np.full((10, 10), 20.0, dtype=np.float32)
    cube = np.stack([b1, b2], axis=0)
    reader = MemoryRasterReader(cube, name="Cube")

    # Square polygon from (2, 2) to (6, 6)
    pts = [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)]
    roi = ROI(roi_id="poly_sq", name="Square ROI", polygon_points=pts)

    stats = roi.calculate_statistics(b1)
    assert stats["count"] > 0
    np.testing.assert_allclose(stats["mean"], 10.0, rtol=1e-4)

    mean_spectrum = roi.calculate_mean_spectrum(reader)
    assert mean_spectrum is not None
    assert len(mean_spectrum) == 2
    np.testing.assert_allclose(mean_spectrum, [10.0, 20.0], rtol=1e-4)
