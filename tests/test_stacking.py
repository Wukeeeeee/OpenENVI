"""Unit tests for layer stacking engine."""

import numpy as np
import pytest
from core.algorithms.stacking import stack_bands
from core.io.memory import MemoryRasterReader


def test_stack_identical_dimensions():
    b1 = np.ones((20, 30), dtype=np.float32) * 1.0
    b2 = np.ones((20, 30), dtype=np.float32) * 2.0
    r1 = MemoryRasterReader(b1, name="Layer1")
    r2 = MemoryRasterReader(b2, name="Layer2")

    progress = []
    cube, meta = stack_bands(
        [(r1, 0), (r2, 0)],
        progress_callback=lambda c, t: progress.append((c, t)),
    )

    assert cube.shape == (20, 30, 2)
    assert meta.bands == 2
    assert np.all(cube[:, :, 0] == 1.0)
    assert np.all(cube[:, :, 1] == 2.0)
    assert progress == [(1, 2), (2, 2)]


def test_stack_different_dimensions_resamples():
    b1 = np.ones((20, 30), dtype=np.float32) * 5.0
    b2 = np.ones((10, 15), dtype=np.float32) * 10.0
    r1 = MemoryRasterReader(b1, name="RefLayer")
    r2 = MemoryRasterReader(b2, name="SmallLayer")

    cube, meta = stack_bands([(r1, 0), (r2, 0)])

    assert cube.shape == (20, 30, 2)
    assert meta.bands == 2
    assert np.allclose(cube[:, :, 0], 5.0)
    assert np.allclose(cube[:, :, 1], 10.0)
