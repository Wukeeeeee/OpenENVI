"""Tests for OpenENVI Algorithms, Spectral Analysis, and ROI Tools.

Verifies stretch modes, spectral indices, Band Math AST evaluation,
PCA, MNF, SAM, K-Means, ISODATA, and ROI metrics.
"""

import numpy as np
import pytest

from core.algorithms.classification import (
    create_thematic_rgb,
    isodata_clustering,
    kmeans_clustering,
    maximum_likelihood_classification,
)
from core.algorithms.indices import (
    calculate_evi,
    calculate_nbr,
    calculate_ndvi,
    calculate_ndwi,
    calculate_savi,
    evaluate_band_math,
)
from core.algorithms.spectral import (
    compute_mnf,
    compute_pca,
    spectral_angle_mapper,
)
from core.algorithms.stretch import (
    apply_stretch,
    gaussian_stretch,
    histogram_equalization_stretch,
    linear_percent_stretch,
    min_max_stretch,
)
from core.roi import ROI
from core.synthetic import generate_synthetic_cube


def test_contrast_stretches():
    """Verify all contrast enhancement algorithms on 2D and 3D arrays."""
    data_2d = np.linspace(10, 500, 100).reshape(10, 10).astype(np.float32)

    # Linear percent stretch
    s_lin = linear_percent_stretch(data_2d, percent=2.0)
    assert s_lin.min() == pytest.approx(0.0)
    assert s_lin.max() == pytest.approx(1.0)
    assert s_lin.shape == (10, 10)

    # Histogram equalization
    s_eq = histogram_equalization_stretch(data_2d)
    assert s_eq.min() >= 0.0
    assert s_eq.max() <= 1.0

    # Gaussian stretch
    s_gauss = gaussian_stretch(data_2d)
    assert s_gauss.min() >= 0.0
    assert s_gauss.max() <= 1.0

    # Min-max stretch
    s_mm = min_max_stretch(data_2d)
    assert s_mm[0, 0] == pytest.approx(0.0)
    assert s_mm[-1, -1] == pytest.approx(1.0)

    # Apply stretch dispatcher with 3D RGB array
    data_3d = np.random.rand(16, 16, 3).astype(np.float32) * 1000
    u8_res = apply_stretch(data_3d, mode="Linear 2%")
    assert u8_res.dtype == np.uint8
    assert u8_res.shape == (16, 16, 3)


def test_spectral_indices():
    """Verify vegetation, water, and soil index calculations."""
    nir = np.full((10, 10), 0.8, dtype=np.float32)
    red = np.full((10, 10), 0.1, dtype=np.float32)
    green = np.full((10, 10), 0.3, dtype=np.float32)
    blue = np.full((10, 10), 0.05, dtype=np.float32)
    swir2 = np.full((10, 10), 0.05, dtype=np.float32)

    ndvi = calculate_ndvi(nir, red)
    assert ndvi.shape == (10, 10)
    # NDVI = (0.8 - 0.1) / (0.8 + 0.1) = 0.7 / 0.9 = 0.777...
    assert ndvi[0, 0] == pytest.approx(0.7777, abs=1e-3)

    ndwi = calculate_ndwi(green, nir)
    assert ndwi.shape == (10, 10)

    evi = calculate_evi(nir, red, blue)
    assert evi.shape == (10, 10)

    savi = calculate_savi(nir, red)
    assert savi.shape == (10, 10)

    nbr = calculate_nbr(nir, swir2)
    assert nbr.shape == (10, 10)


def test_band_math_ast():
    """Verify AST-based Band Math evaluator with expressions and math functions."""
    b1 = np.ones((5, 5), dtype=np.float32) * 10.0
    b2 = np.ones((5, 5), dtype=np.float32) * 2.0
    b3 = np.ones((5, 5), dtype=np.float32) * 5.0

    vars_dict = {"B1": b1, "B2": b2, "b3": b3}

    res1 = evaluate_band_math("(b1 - b2) / b3", vars_dict)
    # (10 - 2) / 5 = 1.6
    np.testing.assert_allclose(res1, 1.6, rtol=1e-5)

    res2 = evaluate_band_math("sqrt(b1 * 10.0) + exp(0.0)", vars_dict)
    # sqrt(100) + 1 = 11.0
    np.testing.assert_allclose(res2, 11.0, rtol=1e-5)

    # Reduction & element-wise min/max
    res_min1 = evaluate_band_math("min(b1)", vars_dict)
    assert float(res_min1[0, 0]) == 10.0
    res_max2 = evaluate_band_math("max(b1, b2)", vars_dict)
    np.testing.assert_allclose(res_max2, 10.0)

    # Normalization formula: (b1 - min(b1)) / (max(b1) - min(b1))
    ramp = np.arange(25, dtype=np.float32).reshape((5, 5))
    res_norm = evaluate_band_math("(ramp - min(ramp)) / (max(ramp) - min(ramp))", {"ramp": ramp})
    assert res_norm.min() == pytest.approx(0.0)
    assert res_norm.max() == pytest.approx(1.0)

    # Mean and Where
    res_mean = evaluate_band_math("mean(ramp)", {"ramp": ramp})
    assert float(res_mean[0, 0]) == pytest.approx(12.0)

    res_where = evaluate_band_math("where(ramp > 12.0, 1.0, 0.0)", {"ramp": ramp})
    assert (res_where > 0).sum() == 12


def test_pca_and_mnf():
    """Verify PCA and MNF dimensional reduction on synthetic cube."""
    cube, wl, gt = generate_synthetic_cube(lines=32, samples=32, bands=16)

    # PCA
    pca_scores, e_vals, exp_var = compute_pca(cube, num_components=3)
    assert pca_scores.shape == (3, 32, 32)
    assert len(e_vals) == 16
    assert len(exp_var) == 3
    # Top component should explain most variance
    assert exp_var[0] > exp_var[1]

    # MNF
    mnf_scores, mnf_eigs = compute_mnf(cube, num_components=3)
    assert mnf_scores.shape == (3, 32, 32)
    assert len(mnf_eigs) == 3


def test_spectral_angle_mapper():
    """Verify SAM classification against known endmembers."""
    cube, wl, gt = generate_synthetic_cube(lines=24, samples=24, bands=12)

    # Extract 2 reference spectra (one from top-left, one from top-right)
    ref_veg = cube[:, 2, 2]
    ref_water = cube[:, 2, 20]
    refs = np.vstack([ref_veg, ref_water])

    rules, cmap = spectral_angle_mapper(cube, refs)
    assert rules.shape == (2, 24, 24)
    assert cmap.shape == (24, 24)

    # Top-left pixel should map to class 0 (veg)
    assert cmap[2, 2] == 0
    # Top-right pixel should map to class 1 (water)
    assert cmap[2, 20] == 1


def test_clustering_and_thematic():
    """Verify K-Means, ISODATA, and thematic RGB mapping."""
    cube, wl, gt = generate_synthetic_cube(lines=30, samples=30, bands=8)

    # K-Means
    km_map, centers = kmeans_clustering(cube, num_classes=4, max_iter=10)
    assert km_map.shape == (30, 30)
    assert centers.shape == (4, 8)
    assert len(np.unique(km_map)) <= 4

    # ISODATA
    iso_map, iso_centers = isodata_clustering(cube, initial_classes=4, max_iter=5)
    assert iso_map.shape == (30, 30)

    # Thematic RGB
    thematic_rgb = create_thematic_rgb(km_map)
    assert thematic_rgb.shape == (30, 30, 3)
    assert thematic_rgb.dtype == np.uint8


def test_roi_statistics():
    """Verify ROI mask creation and statistical extraction."""
    roi = ROI(roi_id="roi_1", name="Forest", bbox=(2, 2, 8, 8))
    band = np.ones((10, 10), dtype=np.float32) * 5.0

    mask = roi.get_mask(10, 10)
    assert mask.shape == (10, 10)
    assert np.sum(mask) == 36  # 6 x 6

    stats = roi.calculate_statistics(band)
    assert stats["count"] == 36
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["min"] == pytest.approx(5.0)
    assert stats["max"] == pytest.approx(5.0)
    assert stats["std"] == pytest.approx(0.0)


def test_maximum_likelihood_classification():
    """Verify supervised Maximum Likelihood Classification on distinct classes."""
    bands, lines, samples = 4, 20, 20
    cube = np.zeros((bands, lines, samples), dtype=np.float32)

    # Class 0: Left half is centered around [10, 20, 30, 40]
    cube[:, :, :10] = np.array([10, 20, 30, 40], dtype=np.float32)[:, None, None]
    cube[:, :, :10] += np.random.normal(0, 0.5, (bands, lines, 10)).astype(np.float32)

    # Class 1: Right half is centered around [80, 70, 60, 50]
    cube[:, :, 10:] = np.array([80, 70, 60, 50], dtype=np.float32)[:, None, None]
    cube[:, :, 10:] += np.random.normal(0, 0.5, (bands, lines, 10)).astype(np.float32)

    # Training samples from each half
    train_c0 = cube[:, :5, :5].reshape(bands, -1).T
    train_c1 = cube[:, :5, 15:].reshape(bands, -1).T

    training_data = {0: train_c0, 1: train_c1}

    class_map, dists = maximum_likelihood_classification(cube, training_data, probability_threshold=0.0)

    assert class_map.shape == (20, 20)
    assert dists.shape == (2, 20, 20)

    # Left half should be class 0, right half class 1 with > 98% accuracy
    assert (class_map[:, :10] == 0).sum() >= 195
    assert (class_map[:, 10:] == 1).sum() >= 195

    # Test probability threshold: an extreme outlier pixel should be classified as -1
    cube[:, 0, 0] = 9999.0
    class_map_thresh, _ = maximum_likelihood_classification(
        cube, training_data, probability_threshold=0.01
    )
    assert class_map_thresh[0, 0] == -1
