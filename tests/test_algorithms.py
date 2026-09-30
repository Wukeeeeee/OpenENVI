"""Tests for OpenENVI Algorithms, Spectral Analysis, and ROI Tools.

Verifies stretch modes, spectral indices, Band Math AST evaluation,
PCA, MNF, ICA, SAM, SID, K-Means, ISODATA, SVM, and ROI metrics.
"""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from core.algorithms.classification import (
    create_thematic_rgb,
    isodata_clustering,
    kmeans_clustering,
    maximum_likelihood_classification,
    svm_classification,
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
    compute_ica,
    compute_mnf,
    compute_pca,
    spectral_angle_mapper,
    spectral_information_divergence,
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


def _two_class_cube(bands=4, lines=20, samples=20, seed=7):
    """Build a noisy two-class cube: left half near [10,20,30,40], right near [80,70,60,50]."""
    rng = np.random.default_rng(seed)
    cube = np.zeros((bands, lines, samples), dtype=np.float32)
    cube[:, :, : samples // 2] = np.array([10, 20, 30, 40], dtype=np.float32)[:, None, None]
    cube[:, :, : samples // 2] += rng.normal(0, 0.5, (bands, lines, samples // 2)).astype(np.float32)
    cube[:, :, samples // 2 :] = np.array([80, 70, 60, 50], dtype=np.float32)[:, None, None]
    cube[:, :, samples // 2 :] += rng.normal(0, 0.5, (bands, lines, samples - samples // 2)).astype(np.float32)
    return cube


def test_svm_classification():
    """Verify supervised SVM classification separates well-separated classes."""
    cube = _two_class_cube()

    training_data = {
        0: cube[:, :5, :5].reshape(4, -1).T,
        1: cube[:, :5, 15:].reshape(4, -1).T,
    }

    progress_calls = []
    class_map, probs = svm_classification(
        cube,
        training_data,
        kernel="rbf",
        C=100.0,
        gamma="scale",
        probability_threshold=0.0,
        progress_callback=lambda pct, msg: progress_calls.append((pct, msg)),
    )

    assert class_map.shape == (20, 20)
    assert probs.shape == (2, 20, 20)

    # Left half class 0, right half class 1, > 98% accuracy on both halves
    assert (class_map[:, :10] == 0).sum() >= 195
    assert (class_map[:, 10:] == 1).sum() >= 195

    # Probabilities are proper distributions
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-4)

    # Progress callback must reach 95 and be monotonically non-decreasing
    assert progress_calls
    assert progress_calls[-1][0] == 95
    pcts = [p for p, _ in progress_calls]
    assert pcts == sorted(pcts)


def test_svm_classification_threshold_and_guards():
    """Verify probability rejection, NaN handling, and single-class guard."""
    cube = _two_class_cube()
    training_data = {
        0: cube[:, :5, :5].reshape(4, -1).T,
        1: cube[:, :5, 15:].reshape(4, -1).T,
    }

    # NaN pixel must become unclassified and never raise
    cube_nan = cube.copy()
    cube_nan[:, 0, 0] = np.nan
    class_map, _ = svm_classification(cube_nan, training_data, probability_threshold=0.0)
    assert class_map[0, 0] == -1

    # An aggressive probability threshold masks pixels as unclassified
    strict_map, _ = svm_classification(cube_nan, training_data, probability_threshold=0.999999)
    assert (strict_map == -1).sum() > 0

    # A single training class is rejected
    with pytest.raises(ValueError):
        svm_classification(cube, {0: training_data[0]})


def test_svm_classification_sample_cap():
    """Verify the per-class training cap subsamples without changing the outcome shape."""
    cube = _two_class_cube()

    # Oversized class to exercise the cap: 200 pixels against a cap of 20
    big_c0 = cube[:, :, :10].reshape(4, -1).T
    assert len(big_c0) > 20
    training_data = {0: big_c0, 1: cube[:, :5, 15:].reshape(4, -1).T}

    capped_map, capped_probs = svm_classification(
        cube, training_data, max_samples_per_class=20
    )
    full_map, full_probs = svm_classification(
        cube, training_data, max_samples_per_class=0
    )

    assert capped_map.shape == full_map.shape == (20, 20)
    assert capped_probs.shape == full_probs.shape

    # Subsampling must preserve separability on well-separated classes
    assert (capped_map[:, :10] == 0).sum() >= 195
    assert (capped_map[:, 10:] == 1).sum() >= 195

    # The two must agree on the overwhelming majority of pixels
    assert (capped_map == full_map).mean() > 0.9


def test_spectral_information_divergence():
    """Verify SID assigns pixels to the nearest reference spectrum."""
    cube = _two_class_cube()

    refs = np.vstack([cube[:, 2, 2], cube[:, 2, 18]])
    rules, class_map = spectral_information_divergence(cube, refs)

    assert rules.shape == (2, 20, 20)
    assert class_map.shape == (20, 20)

    # Divergence is symmetric and non-negative
    assert np.all(rules >= 0.0)

    # Left half should be class 0, right half class 1
    assert (class_map[:, :10] == 0).sum() >= 195
    assert (class_map[:, 10:] == 1).sum() >= 195

    # A pixel matching a reference spectrum exactly has ~zero divergence
    rules_exact, class_map_exact = spectral_information_divergence(
        cube[:, 2:3, 2:3], refs
    )
    assert float(np.min(rules_exact)) == pytest.approx(0.0, abs=1e-5)
    assert class_map_exact[0, 0] == 0

    # NaN pixels are masked as unclassified
    cube_nan = cube.copy()
    cube_nan[:, 0, 0] = np.nan
    _, cmap_nan = spectral_information_divergence(cube_nan, refs)
    assert cmap_nan[0, 0] == -1


def test_spectral_information_divergence_threshold():
    """Verify the maximum divergence threshold masks dissimilar pixels."""
    cube = _two_class_cube()
    refs = np.vstack([cube[:, 2, 2], cube[:, 2, 18]])

    # SID normalizes spectra to probability vectors, so it is invariant to overall
    # brightness. The noisy majority therefore has small-but-nonzero divergence.
    _, strict_map = spectral_information_divergence(cube, refs, max_divergence=1e-6)
    assert (strict_map == -1).sum() > 0.95 * strict_map.size

    # A loose threshold keeps every pixel classified
    _, loose_map = spectral_information_divergence(cube, refs, max_divergence=1e9)
    assert not (loose_map == -1).any()


def test_compute_ica():
    """Verify FastICA separates two statistically independent mixed sources."""
    rng = np.random.default_rng(3)
    lines = samples = 30
    num_pixels = lines * samples

    # Two independent sources: one uniform, one strongly peaked (non-Gaussian)
    src_a = rng.random(num_pixels).astype(np.float32)
    src_b = rng.gamma(shape=0.6, scale=2.0, size=num_pixels).astype(np.float32)

    # Mix them with a fixed 4x2 matrix so the cube has 4 bands and 2 components
    mixing = np.array(
        [[1.0, 0.5], [0.3, 1.2], [0.8, 0.2], [0.4, 0.9]], dtype=np.float32
    )
    mixed = (mixing @ np.vstack([src_a, src_b])).astype(np.float32)  # (4, num_pixels)
    cube = mixed.reshape(4, lines, samples)

    ica_cube, mixing_mat = compute_ica(cube, num_components=2, random_state=42)

    assert ica_cube.shape == (2, lines, samples)
    assert mixing_mat.shape == (4, 2)
    assert np.all(np.isfinite(ica_cube))

    # Each recovered component must be uncorrelated with the other
    centered = ica_cube.reshape(2, -1).astype(np.float64)
    centered -= centered.mean(axis=1, keepdims=True)
    corr = np.corrcoef(centered)
    assert abs(corr[0, 1]) < 0.2

    # Requesting more components than bands is clamped to the band count. Decomposing
    # a 4-band cube into all 4 components is a degenerate case for FastICA, which may
    # emit a ConvergenceWarning; only the shape clamping is under test here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clamped, clamped_mixing = compute_ica(cube, num_components=99)
    assert clamped.shape[0] == 4
    assert clamped_mixing.shape == (4, 4)


def test_mnf_whitens_noise():
    """MNF components must carry unit-variance (whitened) noise, not the sensor covariance."""
    rng = np.random.default_rng(0)
    # Spatially white noise with strongly unequal per-band variances
    var = np.array([1.0, 5.0, 20.0, 100.0])
    cube = (rng.standard_normal((4, 200, 200)) * np.sqrt(var)[:, None, None]).astype(np.float32)
    cube += np.array([50.0, 20.0, 10.0, 5.0], dtype=np.float32)[:, None, None]

    mnf, eigenvalues = compute_mnf(cube, num_components=3)
    assert mnf.shape == (3, 200, 200)

    # Residual noise estimated from horizontal differences (the mean offset cancels).
    # For whitened components this equals 2 * 1.0 = 2.0, since Var(x[i+1]-x[i]) = 2*Var.
    diffs = np.stack([(mnf[k, :, 1:] - mnf[k, :, :-1]).ravel() for k in range(3)])
    noise_var = np.diag(np.cov(diffs))
    np.testing.assert_allclose(noise_var, 2.0, rtol=0.05)


def test_transforms_tolerate_nan_pixels():
    """PCA, MNF and ICA must not fail on cubes containing NoData (NaN) pixels."""
    rng = np.random.default_rng(1)
    cube = rng.random((6, 20, 20)).astype(np.float32)
    cube[3, 7, 9] = np.nan

    scores, eigvals, evr = compute_pca(cube, num_components=3)
    assert scores.shape == (3, 20, 20)
    assert np.all(np.isfinite(scores[:, :7, :])) and np.all(np.isfinite(scores[:, 8:, :]))

    mnf, mnf_eig = compute_mnf(cube, num_components=3)
    assert mnf.shape == (3, 20, 20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ica, mixing = compute_ica(cube, num_components=3, max_iter=500)
    assert ica.shape == (3, 20, 20)
    assert mixing.shape == (6, 3)
    # The NaN pixel stays NaN rather than contaminating the whole component
    assert np.all(np.isnan(ica[:, 7, 9]))
    assert np.all(np.isfinite(ica[:, 0, 0]))


def test_roi_statistics_exclude_nodata():
    """ROI statistics must ignore finite NoData sentinels, not just NaN/Inf."""
    roi = ROI(roi_id="r", name="x", bbox=(0, 0, 4, 4))

    band = np.full((10, 10), 0.5, dtype=np.float32)
    band[0:2, 0:2] = -9999.0

    # Without the sentinel hint the raw values are reported (documented behaviour)
    legacy = roi.calculate_statistics(band)
    assert legacy["min"] == pytest.approx(-9999.0)

    # With it, the NoData block is excluded
    stats = roi.calculate_statistics(band, nodata=-9999.0)
    assert stats["count"] == 12
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["min"] == pytest.approx(0.5)
    assert stats["max"] == pytest.approx(0.5)


def test_roi_mean_spectrum_excludes_nodata():
    """ROI mean spectra (the SAM/SID/SVM endmember source) must skip NoData pixels."""
    from core.io.memory import MemoryRasterReader
    from core.models import BandInfo, RasterMetadata

    cube = np.full((2, 10, 10), 100.0, dtype=np.float32)
    cube[0, 0:2, 0:2] = -9999.0  # NoData block inside the ROI
    meta = RasterMetadata(
        width=10, height=10, bands=2, dtype=np.float32, nodata=-9999.0,
        band_details=[BandInfo(index=0, name="A"), BandInfo(index=1, name="B")],
    )
    reader = MemoryRasterReader(cube, name="nd", parent_metadata=meta)

    roi = ROI(roi_id="r", name="x", bbox=(0, 0, 4, 4))
    spectrum = roi.calculate_mean_spectrum(reader)
    # Band 0: 12 valid pixels at 100 plus a 2x2 NoData block that must be skipped

    assert spectrum is not None
    # Band 0 is contaminated by NoData but must still average to 100
    np.testing.assert_allclose(spectrum, [100.0, 100.0], rtol=1e-5)
