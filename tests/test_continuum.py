"""Unit tests for continuum removal engine."""

import numpy as np
import pytest
from core.algorithms.continuum import continuum_removal_1d, continuum_removal_cube


def test_continuum_removal_1d():
    # Spectrum with absorption pit at index 3
    spec = np.array([0.5, 0.8, 0.4, 0.3, 0.7, 0.9, 0.6], dtype=np.float32)
    wl = np.array([400, 500, 600, 700, 800, 900, 1000], dtype=np.float32)

    cr, cont = continuum_removal_1d(spec, wl)
    assert len(cr) == 7
    assert len(cont) == 7
    # Peaks should equal 1.0
    assert np.isclose(cr[1], 1.0)
    assert np.isclose(cr[5], 1.0)
    # Absorption minimum should be strictly < 1.0
    assert cr[3] < 1.0
    # Values should never exceed 1.0
    assert np.all(cr <= 1.0)
    assert np.all(cr >= 0.0)


def test_continuum_removal_cube():
    cube = np.random.rand(4, 5, 6).astype(np.float32) + 0.1
    progress = []
    cr_cube = continuum_removal_cube(cube, progress_callback=lambda c, t: progress.append((c, t)))

    assert cr_cube.shape == (4, 5, 6)
    assert np.all(cr_cube <= 1.0)
    assert np.all(cr_cube >= 0.0)
    assert len(progress) == 4
