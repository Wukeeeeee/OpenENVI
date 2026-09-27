"""Unit tests for raster statistics calculation."""

import numpy as np
from core.algorithms.statistics import calculate_band_statistics, calculate_raster_statistics
from core.io.memory import MemoryRasterReader


def test_calculate_band_statistics():
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float32)

    stats = calculate_band_statistics(data, bins=10)
    assert stats["count"] == 9
    assert stats["min"] == 1.0
    assert stats["max"] == 9.0
    assert np.isclose(stats["mean"], 5.0)
    assert len(stats["hist_counts"]) == 10
    assert np.sum(stats["hist_counts"]) == 9


def test_calculate_raster_statistics_streaming():
    b1 = np.ones((5, 5), dtype=np.float32) * 10.0
    b2 = np.ones((5, 5), dtype=np.float32) * 20.0
    cube = np.stack([b1, b2], axis=0)

    reader = MemoryRasterReader(cube, name="Test Cube")
    progress = []
    res = calculate_raster_statistics(reader, progress_callback=lambda c, t: progress.append((c, t)))

    assert len(res) == 2
    assert res[0]["mean"] == 10.0
    assert res[1]["mean"] == 20.0
    assert progress == [(1, 2), (2, 2)]
