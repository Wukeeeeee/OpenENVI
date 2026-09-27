"""Unit tests for Pan-Sharpening algorithms (Gram-Schmidt & Brovey)."""

import pytest
import numpy as np

from core.algorithms.pansharpen import (
    brovey_pansharpen,
    gram_schmidt_pansharpen,
    resample_band_to_grid,
)


def test_resample_band_to_grid():
    """Test bilinear resampling from low-res to high-res grid."""
    low_res = np.array([[10, 20], [30, 40]], dtype=np.float32)
    high_res = resample_band_to_grid(low_res, target_height=4, target_width=4)
    assert high_res.shape == (4, 4)
    assert high_res.dtype == np.float32
    # Ensure values interpolate within range [10, 40]
    assert high_res.min() >= 10.0 - 1e-4
    assert high_res.max() <= 40.0 + 1e-4


def test_brovey_pansharpen():
    """Test Brovey pan-sharpening with synthetic RGB and Pan."""
    pan = np.full((20, 20), 100.0, dtype=np.float32)
    b_r = np.full((10, 10), 40.0, dtype=np.float32)
    b_g = np.full((10, 10), 50.0, dtype=np.float32)
    b_b = np.full((10, 10), 60.0, dtype=np.float32)

    progress = []
    fused = brovey_pansharpen(
        pan_band=pan,
        ms_bands=[b_r, b_g, b_b],
        progress_callback=lambda s, t: progress.append((s, t)),
    )

    assert fused.shape == (20, 20, 3)
    assert len(progress) > 0
    # Mean of (40, 50, 60) is 50. Pan is 100. Scale factor should be 2.0.
    # Therefore, fused bands should be approximately 80, 100, 120.
    np.testing.assert_allclose(fused[..., 0], 80.0, rtol=1e-3)
    np.testing.assert_allclose(fused[..., 1], 100.0, rtol=1e-3)
    np.testing.assert_allclose(fused[..., 2], 120.0, rtol=1e-3)


def test_gram_schmidt_pansharpen():
    """Test Gram-Schmidt pan-sharpening with synthetic inputs."""
    np.random.seed(42)
    pan = (np.random.rand(30, 30) * 100).astype(np.float32)
    ms1 = (np.random.rand(15, 15) * 50).astype(np.float32)
    ms2 = (np.random.rand(15, 15) * 50).astype(np.float32)

    progress = []
    fused = gram_schmidt_pansharpen(
        pan_band=pan,
        ms_bands=[ms1, ms2],
        progress_callback=lambda s, t: progress.append((s, t)),
    )

    assert fused.shape == (30, 30, 2)
    assert len(progress) > 0
    assert np.all(np.isfinite(fused))
