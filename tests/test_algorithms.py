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
    spectral_feature_fitting,
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


def _planted_endmember_scene(num_bands: int = 30, n_pixels: int = 20000):
    """Build a cube from four shaped endmember spectra with near-pure pixels.

    Real land cover contains large near-pure regions, and endmember extraction
    depends on it: with dense mixture abundances every pixel sits in the interior
    of the simplex and none of them lies near any candidate ray.
    """
    wl = np.linspace(0.4, 2.4, num_bands)
    truth = np.stack([
        np.exp(-((wl - 1.4) ** 2) / 0.02) + 0.45 * np.exp(-((wl - 0.68) ** 2) / 0.004),
        0.15 + 0.55 * (wl - 0.4) / 2.0 + 0.05 * np.cos(wl * 6),
        0.04 + 0.10 * np.exp(-((wl - 0.44) ** 2) / 0.01) + 0.01 * wl,
        0.30 * np.exp(-((wl - 0.9) ** 2) / 0.08),
    ])
    rng = np.random.default_rng(0)
    labels = rng.choice(4, size=n_pixels, p=[0.35, 0.30, 0.20, 0.15])
    abund = rng.dirichlet(np.ones(4), size=n_pixels) * 0.25
    abund[np.arange(n_pixels), labels] += 0.75
    flat = abund @ truth + 0.01 * np.abs(rng.standard_normal((n_pixels, num_bands)))
    return flat.T.reshape(num_bands, 200, n_pixels // 200), truth


def test_spectral_feature_fitting_recovers_planted_endmembers():
    """SFF must return distinct, accurate endmember spectra, not one ray repeated."""
    cube, truth = _planted_endmember_scene()
    truth_n = truth / np.linalg.norm(truth, axis=1, keepdims=True)

    ends, residuals = spectral_feature_fitting(cube, num_features=3)

    assert ends.shape == (3, truth.shape[1])
    assert residuals.shape == (3,)
    # Reflectance-like spectra: non-negative and finite everywhere
    assert np.all(np.isfinite(ends))
    assert ends.min() >= 0.0

    found_n = ends / np.linalg.norm(ends, axis=1, keepdims=True)
    corr = found_n @ truth_n.T
    matched = [int(np.argmax(c)) for c in corr]
    # Three different planted endmembers, each matched well
    assert len(set(matched)) == 3, matched
    assert corr.max(axis=1).min() > 0.95, corr
    # Recovered spectra must not collapse onto one another
    assert (found_n @ found_n.T)[np.triu_indices(3, 1)].max() < 0.85


def test_spectral_feature_fitting_is_deterministic():
    """The same input and seed must produce the same endmembers."""
    cube, _ = _planted_endmember_scene()
    a, ra = spectral_feature_fitting(cube, num_features=3, seed=7)
    b, rb = spectral_feature_fitting(cube, num_features=3, seed=7)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(ra, rb)


def test_spectral_feature_fitting_auto_count_skips_dead_bands():
    """Auto-selection must not spend endmembers on constant, information-free bands."""
    cube, _ = _planted_endmember_scene()
    dead = cube.copy()
    dead[5] = 1.0    # constant band: zero variance, no endmember information
    dead[11] = 2.5

    ends, _ = spectral_feature_fitting(dead, num_features=0)

    assert len(ends) >= 1
    # A dead band is flat, so it can never carry a shaped endmember spectrum
    for spectrum in ends:
        assert spectrum.std() > 0.0


def test_spectral_feature_fitting_ignores_nan_pixels():
    """Non-finite pixels must be dropped rather than poison the covariance."""
    cube, truth = _planted_endmember_scene()
    truth_n = truth / np.linalg.norm(truth, axis=1, keepdims=True)

    holed = cube.copy()
    holed[:, :20, :20] = np.nan
    holed[:, 5, 5] = np.inf

    ends, _ = spectral_feature_fitting(holed, num_features=3)
    found_n = ends / np.linalg.norm(ends, axis=1, keepdims=True)

    assert np.all(np.isfinite(ends))
    corr = found_n @ truth_n.T
    assert len(set(int(np.argmax(c)) for c in corr)) == 3
    assert corr.max(axis=1).min() > 0.95, corr


def test_spectral_feature_fitting_respects_max_samples():
    """max_samples bounds the search cost without changing the answer's character."""
    cube, truth = _planted_endmember_scene()
    truth_n = truth / np.linalg.norm(truth, axis=1, keepdims=True)

    ends, _ = spectral_feature_fitting(cube, num_features=3, max_samples=3000)
    found_n = ends / np.linalg.norm(ends, axis=1, keepdims=True)
    corr = found_n @ truth_n.T

    assert len(set(int(np.argmax(c)) for c in corr)) == 3
    assert corr.max(axis=1).min() > 0.9, corr


def test_spectral_feature_fitting_requires_enough_pixels():
    """Too few valid pixels for a covariance must be reported, not silently fitted."""
    cube = np.ones((6, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        spectral_feature_fitting(cube)


def test_least_squares_abundances_recover_planted_abundances():
    """Unmixing a scene on its own endmembers must reproduce the abundances."""
    from ui.dialogs.sff_dialog import least_squares_abundances

    wl = np.linspace(0.4, 2.4, 30)
    ends = np.stack([
        np.exp(-((wl - 1.4) ** 2) / 0.02) + 0.45 * np.exp(-((wl - 0.68) ** 2) / 0.004),
        0.15 + 0.55 * (wl - 0.4) / 2.0 + 0.05 * np.cos(wl * 6),
        0.30 * np.exp(-((wl - 0.9) ** 2) / 0.08),
    ])
    rng = np.random.default_rng(3)
    abund = rng.dirichlet(np.ones(3), size=4000) * 0.5
    abund[np.arange(4000), rng.integers(0, 3, 4000)] += 0.5
    flat = abund @ ends
    cube = flat.T.reshape(30, 40, 100)

    recovered = least_squares_abundances(cube, ends)

    assert recovered.shape == (3, 40, 100)
    assert recovered.min() >= 0.0
    got = recovered.reshape(3, -1).T
    np.testing.assert_allclose(got, abund, atol=1e-6)


# ---------------------------------------------------------------------------
# Mosaicking
# ---------------------------------------------------------------------------


def _geo_tile(shape, fill, origin_x, origin_y, crs="EPSG:32650", res=30.0, nodata=None):
    """Single-band georeferenced reader of constant value.

    ``origin_y`` is the y coordinate of the tile's TOP edge, matching the
    north-up convention the GeoTIFF reader writes.
    """
    from core.io.memory import MemoryRasterReader
    from core.models import RasterMetadata

    data = np.full(shape, float(fill), dtype=np.float32)
    meta = RasterMetadata(
        width=shape[1],
        height=shape[0],
        bands=1,
        dtype="float32",
        crs=crs,
        transform=(res, 0.0, origin_x, 0.0, -res, origin_y),
        nodata=nodata,
    )
    return MemoryRasterReader(data, name=f"tile_{origin_x}_{origin_y}", parent_metadata=meta)


def test_build_mosaic_grid_unions_adjacent_tiles():
    """Two touching tiles must produce one grid twice as wide as either."""
    from core.algorithms.mosaic import build_mosaic_grid

    left = _geo_tile((10, 10), 1.0, 500000.0, 4000000.0)
    right = _geo_tile((10, 10), 2.0, 500300.0, 4000000.0)

    transform, width, height, crs = build_mosaic_grid([left, right])

    assert (width, height) == (20, 10)
    assert crs == "EPSG:32650"
    # North-up: the top edge is the union top and row 0 advances southward.
    assert transform.f == pytest.approx(4000000.0)
    assert transform.e < 0
    assert transform.c == pytest.approx(500000.0)


def test_mosaic_adjacent_tiles_keeps_every_value():
    """Side-by-side tiles must land side by side, not stacked or flipped."""
    from core.algorithms.mosaic import mosaic_rasters

    left = _geo_tile((8, 10), 5.0, 500000.0, 4000000.0)
    right = _geo_tile((8, 10), 9.0, 500300.0, 4000000.0)

    cube, meta = mosaic_rasters([left, right], resample_method="nearest")

    assert cube.shape == (8, 20, 1)
    np.testing.assert_allclose(cube[:, :10, 0], 5.0, rtol=1e-5)
    np.testing.assert_allclose(cube[:, 10:, 0], 9.0, rtol=1e-5)
    assert meta.crs == "EPSG:32650"
    assert meta.transform[4] < 0


def test_mosaic_fills_uncovered_area_with_background():
    """A gap between tiles must carry background_value, not NaN."""
    from core.algorithms.mosaic import mosaic_rasters

    left = _geo_tile((6, 6), 3.0, 500000.0, 4000000.0)
    # The second tile starts 12 pixels east, leaving a 6-pixel hole between them.
    right = _geo_tile((6, 6), 7.0, 500360.0, 4000000.0)

    cube, _ = mosaic_rasters([left, right], background_value=-1.0, resample_method="nearest")

    assert cube.shape == (6, 18, 1)
    np.testing.assert_allclose(cube[:, 6:12, 0], -1.0)
    np.testing.assert_allclose(cube[:, :6, 0], 3.0, rtol=1e-5)
    np.testing.assert_allclose(cube[:, 12:, 0], 7.0, rtol=1e-5)


def test_mosaic_feathers_overlapping_tiles():
    """With feathering the overlap becomes a gradient instead of a hard switch."""
    from core.algorithms.mosaic import mosaic_rasters

    left = _geo_tile((20, 20), 0.0, 500000.0, 4000000.0)
    right = _geo_tile((20, 20), 100.0, 500300.0, 4000000.0)

    cube, _ = mosaic_rasters([left, right], feather_pixels=8.0, resample_method="nearest")

    overlap = cube[:, 10:20, 0]
    # Monotonic across the 10-pixel overlap, strictly between the two inputs.
    assert np.all(np.diff(overlap, axis=1) > 0)
    assert overlap.min() > 0.0 and overlap.max() < 100.0
    # Where only one source reaches, that source's value survives intact.
    np.testing.assert_allclose(cube[:, 2, 0], 0.0, rtol=1e-5)
    assert cube[:, 19, 0].min() > 90.0

    # Unweighted averaging makes the whole overlap one constant, so the seam is a
    # step; feathering is what turns it into a ramp.
    hard, _ = mosaic_rasters([left, right], feather_pixels=0.0, resample_method="nearest")
    assert np.ptp(hard[:, 10:20, 0]) == 0.0


def test_mosaic_keeps_multiple_bands_separate():
    """Bands must not bleed into each other when mosaicking."""
    from core.algorithms.mosaic import mosaic_rasters
    from core.io.memory import MemoryRasterReader
    from core.models import RasterMetadata

    def tile(x, first, second):
        data = np.stack(
            [np.full((6, 6), first, np.float32), np.full((6, 6), second, np.float32)],
            axis=-1,
        )
        meta = RasterMetadata(
            width=6, height=6, bands=2, dtype="float32", crs="EPSG:32650",
            transform=(30.0, 0.0, x, 0.0, -30.0, 4000000.0), nodata=None,
        )
        return MemoryRasterReader(data, name=f"multi_{x}", parent_metadata=meta)

    cube, meta = mosaic_rasters([tile(500000.0, 1.0, 2.0), tile(500180.0, 3.0, 4.0)])

    assert cube.shape == (6, 12, 2)
    assert meta.bands == 2
    np.testing.assert_allclose(cube[:, :6, 0], 1.0, rtol=1e-5)
    np.testing.assert_allclose(cube[:, :6, 1], 2.0, rtol=1e-5)
    np.testing.assert_allclose(cube[:, 6:, 0], 3.0, rtol=1e-5)
    np.testing.assert_allclose(cube[:, 6:, 1], 4.0, rtol=1e-5)


def test_mosaic_respects_source_nodata():
    """A NoData source pixel must not bleed its sentinel into the output."""
    from core.algorithms.mosaic import mosaic_rasters
    from core.io.memory import MemoryRasterReader
    from core.models import RasterMetadata

    holed = np.full((6, 6), -9999.0, dtype=np.float32)
    holed[:, :3] = 12.0
    reader = MemoryRasterReader(
        holed, name="holed", parent_metadata=RasterMetadata(
            width=6, height=6, bands=1, dtype="float32", crs="EPSG:32650",
            transform=(30.0, 0.0, 500180.0, 0.0, -30.0, 4000000.0), nodata=-9999.0,
        ),
    )
    good = _geo_tile((6, 6), 4.0, 500000.0, 4000000.0)

    cube, _ = mosaic_rasters([good, reader], resample_method="bilinear")

    assert cube.min() > -100.0, "NoData sentinel leaked into the mosaic"
    np.testing.assert_allclose(cube[:, 6:9, 0], 12.0, rtol=1e-5)


def test_mosaic_reports_progress_per_band():
    """The progress callback must fire once per band and finish at the total."""
    from core.algorithms.mosaic import mosaic_rasters

    left = _geo_tile((6, 6), 1.0, 500000.0, 4000000.0)
    right = _geo_tile((6, 6), 2.0, 500180.0, 4000000.0)
    seen = []

    mosaic_rasters([left, right], progress_callback=lambda c, t: seen.append((c, t)))

    assert seen == [(1, 1)]


def test_mosaic_rejects_mismatched_band_counts():
    """Inputs with different band counts cannot share one output grid."""
    from core.algorithms.mosaic import mosaic_rasters
    from core.io.memory import MemoryRasterReader
    from core.models import RasterMetadata

    def tile(x, bands):
        meta = RasterMetadata(
            width=4, height=4, bands=bands, dtype="float32", crs="EPSG:32650",
            transform=(30.0, 0.0, x, 0.0, -30.0, 4000000.0), nodata=None,
        )
        return MemoryRasterReader(
            np.ones((4, 4, bands), np.float32), name=f"b{bands}_{x}", parent_metadata=meta
        )

    with pytest.raises(ValueError, match="same band count"):
        mosaic_rasters([tile(500000.0, 1), tile(500120.0, 2)])


def test_mosaic_rejects_empty_input_and_unknown_resampling():
    """Empty input and a bogus resampling name must both raise cleanly."""
    from core.algorithms.mosaic import mosaic_rasters

    with pytest.raises(ValueError, match="No input rasters"):
        mosaic_rasters([])

    tile = _geo_tile((4, 4), 1.0, 500000.0, 4000000.0)
    with pytest.raises(ValueError, match="Unknown resampling"):
        mosaic_rasters([tile], resample_method="sinc-magic")


def test_mosaic_accepts_gdal_cubic_spline_spelling():
    """ENVI and GDAL disagree on the name; both spellings must work."""
    from core.algorithms.mosaic import mosaic_rasters

    left = _geo_tile((8, 8), 1.0, 500000.0, 4000000.0)
    right = _geo_tile((8, 8), 2.0, 500240.0, 4000000.0)

    aliased, _ = mosaic_rasters([left, right], resample_method="cubicspline")
    native, _ = mosaic_rasters([left, right], resample_method="cubic_spline")

    np.testing.assert_allclose(aliased, native)


# ---------------------------------------------------------------------------
# Spectral Library
# ---------------------------------------------------------------------------


def test_spectral_library_entries_are_well_formed():
    """Every reference spectrum must sit on the shared grid and be usable."""
    from core.algorithms.spectral_library import (
        LIBRARY_WAVELENGTHS,
        find_spectrum,
        get_categories,
        get_spectral_library,
    )

    entries = get_spectral_library()
    assert len(entries) >= 15
    assert len(get_categories()) >= 4

    names = {e.name for e in entries}
    assert len(names) == len(entries), "library contains duplicate names"

    for entry in entries:
        assert entry.reflectance.shape == LIBRARY_WAVELENGTHS.shape
        assert np.all(np.isfinite(entry.reflectance))
        assert entry.reflectance.min() >= 0.0
        # A reflectance library is meaningless above 100 percent.
        assert entry.reflectance.max() <= 1.0
        assert entry.reflectance.mean() > 0.0
        assert LIBRARY_WAVELENGTHS[0] <= entry.peak_wavelength <= LIBRARY_WAVELENGTHS[-1]

    assert find_spectrum("Green Vegetation") is not None
    assert find_spectrum("No Such Spectrum") is None


def test_spectral_library_vegetation_shows_chlorophyll_and_red_edge():
    """Green vegetation must have a red trough, a green peak and a NIR plateau."""
    from core.algorithms.spectral_library import find_spectrum

    veg = find_spectrum("Green Vegetation")
    at = lambda w: float(np.interp(w, veg.wavelengths, veg.reflectance))

    # Red absorption is the deepest point of the visible range.
    assert at(650.0) < at(550.0)
    # The red edge climbs steeply between 680 and 780 nm.
    assert at(760.0) > at(690.0)
    # Leaf water absorbs in the SWIR, below the NIR plateau.
    assert at(1400.0) < 0.6 * at(1100.0)
    assert at(1900.0) < 0.6 * at(1100.0)


def test_spectral_library_water_absorbs_in_the_nir():
    """Water must be near-zero beyond the visible, which is what makes it separable."""
    from core.algorithms.spectral_library import find_spectrum

    water = find_spectrum("Clear Water")
    at = lambda w: float(np.interp(w, water.wavelengths, water.reflectance))

    assert at(450.0) > 0.0
    assert at(1200.0) < 0.01
    assert at(1900.0) < 0.005
    # Turbid water carries more energy into the NIR than clear water.
    turbid = find_spectrum("Turbid Water")
    assert float(np.interp(1000.0, turbid.wavelengths, turbid.reflectance)) > at(1000.0)


def test_resample_spectrum_marks_out_of_range_as_nan():
    """A narrow-band sensor must not be compared against wavelengths it never saw."""
    from core.algorithms.spectral_library import resample_spectrum

    wl = np.array([450.0, 550.0, 650.0])
    vals = np.array([0.1, 0.3, 0.1])
    targets = np.array([400.0, 500.0, 550.0, 800.0])

    out = resample_spectrum(wl, vals, targets)

    assert np.isnan(out[0]), "below range must be NaN"
    assert np.isnan(out[3]), "above range must be NaN"
    assert out[2] == pytest.approx(0.3)
    # 500 nm is the midpoint of 450 and 550.
    assert out[1] == pytest.approx(0.2)


def test_mean_spectrum_ignores_invalid_pixels():
    """NaN pixels must be excluded rather than poisoning the whole average."""
    from core.algorithms.spectral_library import mean_spectrum

    wl = np.array([400.0, 500.0, 600.0])
    cube = np.full((3, 2, 2), 4.0, dtype=np.float32)
    cube[:, 0, 1] = np.nan          # one dead pixel
    cube[:, 1, :] = 6.0             # one brighter line

    curve, count = mean_spectrum(cube, wl)

    assert count == 3
    assert curve.shape == (3,)
    # The three survivors are 4, 6 and 6; the dead pixel contributes nothing.
    assert curve[0] == pytest.approx(16.0 / 3.0)


def test_mean_spectrum_validates_its_inputs():
    """Wrong rank or a band/wavelength mismatch must be reported, not broadcast."""
    from core.algorithms.spectral_library import mean_spectrum

    with pytest.raises(ValueError, match="bands, lines, samples"):
        mean_spectrum(np.ones((5, 5)), [400.0, 500.0])
    with pytest.raises(ValueError, match="wavelengths"):
        mean_spectrum(np.ones((3, 2, 2)), [400.0, 500.0])
    with pytest.raises(ValueError, match="no valid pixels"):
        mean_spectrum(np.full((3, 2, 2), np.nan), [400.0, 500.0, 600.0])


def test_spectral_angle_is_scale_invariant():
    """A pure brightness change must not change the spectral angle."""
    from core.algorithms.spectral_library import spectral_angle_degrees

    a = np.array([0.1, 0.3, 0.5, 0.2])
    b = np.array([0.2, 0.6, 1.0, 0.4])

    assert spectral_angle_degrees(a, a) == pytest.approx(0.0, abs=1e-6)
    assert spectral_angle_degrees(a, b) == pytest.approx(0.0, abs=1e-6)
    # A different shape, however faint, must register.
    assert spectral_angle_degrees(a, np.array([0.5, 0.3, 0.1, 0.2])) > 30.0
    # Identical curves still match where the other one has no data.
    assert spectral_angle_degrees(
        a, np.array([0.1, 0.3, 0.5, np.nan])
    ) == pytest.approx(0.0, abs=1e-6)


def test_match_library_ranks_the_true_material_first():
    """Feeding a library curve back in must return that same curve as the best match."""
    from core.algorithms.spectral_library import get_spectral_library, match_library

    for entry in get_spectral_library():
        results = match_library(entry.wavelengths, entry.reflectance)
        assert results, f"no match returned for {entry.name}"
        best, angle = results[0]
        assert best.name == entry.name
        assert angle < 1e-3

        angles = [a for _, a in results]
        assert angles == sorted(angles), "results must come back ranked by angle"


def test_match_library_filters_by_angle_and_category():
    """The threshold and category filter must both actually remove entries."""
    from core.algorithms.spectral_library import find_spectrum, match_library

    grass = find_spectrum("Dry Grass")
    strict = match_library(grass.wavelengths, grass.reflectance, max_angle=1.0)
    assert len(strict) == 1
    assert strict[0][0].name == "Dry Grass"

    only_rock = match_library(
        grass.wavelengths, grass.reflectance, max_angle=180.0, categories=["Rock"]
    )
    assert only_rock
    assert all(entry.category == "Rock" for entry, _ in only_rock)


def test_match_library_rejects_disjoint_range():
    """A sensor band set outside the library range has nothing to match against."""
    from core.algorithms.spectral_library import match_library

    with pytest.raises(ValueError, match="does not overlap"):
        match_library(
            np.array([20000.0, 21000.0]),
            np.array([0.1, 0.2]),
        )


def test_layer_wavelengths_falls_back_when_the_file_has_none():
    """An image with no wavelength tags must still get a usable matching axis.

    GeoTIFFs without a WAVELENGTH tag report None per band. Both that and a
    malformed NaN have to fall back, or the whole library comparison silently
    collapses onto a single axis point.
    """
    from core.io.memory import MemoryRasterReader
    from core.algorithms.spectral_library import LIBRARY_WAVELENGTHS
    from core.models import BandInfo, RasterMetadata
    from ui.dialogs.spectral_library_dialog import layer_wavelengths

    def reader_with(details):
        meta = RasterMetadata(
            width=4, height=4, bands=3, dtype="float32", crs=None, nodata=None,
            band_details=details,
        )
        return MemoryRasterReader(
            np.ones((4, 4, 3), np.float32), name="wl", parent_metadata=meta
        )

    # Real header wavelengths are used as-is.
    real = reader_with([
        BandInfo(index=0, name="B1", wavelength=460.0),
        BandInfo(index=1, name="B2", wavelength=560.0),
        BandInfo(index=2, name="B3", wavelength=660.0),
    ])
    np.testing.assert_allclose(layer_wavelengths(real, 3), [460.0, 560.0, 660.0])

    # None (no tag) and NaN (bad tag) both fall back across the library range.
    for bad in (None, float("nan")):
        broken = reader_with([
            BandInfo(index=0, name="B1", wavelength=bad),
            BandInfo(index=1, name="B2", wavelength=bad),
            BandInfo(index=2, name="B3", wavelength=bad),
        ])
        axis = layer_wavelengths(broken, 3)
        assert np.all(np.isfinite(axis))
        assert axis[0] == pytest.approx(LIBRARY_WAVELENGTHS[0])
        assert axis[-1] == pytest.approx(LIBRARY_WAVELENGTHS[-1])
        assert np.all(np.diff(axis) > 0)

    # A partially populated header is not usable either: one bad band poisons the axis.
    partial = reader_with([
        BandInfo(index=0, name="B1", wavelength=460.0),
        BandInfo(index=1, name="B2", wavelength=None),
        BandInfo(index=2, name="B3", wavelength=660.0),
    ])
    assert np.all(np.isfinite(layer_wavelengths(partial, 3)))


# ---------------------------------------------------------------------------
# Regressions: ENVI georeferencing, nodata propagation and numeric robustness
# ---------------------------------------------------------------------------


def _write_envi_header(tmp_path, name="t", extra=""):
    """Write a minimal, standards-conforming ENVI dataset and return its path.

    The map info uses a NEGATIVE y-scale, which is what real ENVI emits for a
    north-referenced image.
    """
    import os

    lines, samples, bands = 6, 5, 2
    data = np.arange(bands * lines * samples, dtype=np.int16)
    data.tofile(os.path.join(tmp_path, f"{name}.dat"))
    hdr = (
        "ENVI\n"
        f"samples = {samples}\n"
        f"lines = {lines}\n"
        f"bands = {bands}\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 12\n"
        "interleave = bsq\n"
        "map info = {UTM, 1.000, 1.000, 500000.0, 4000000.0, 30.0, -30.0, "
        "1, North, WGS-84}\n"
        f"{extra}"
    )
    path = os.path.join(tmp_path, f"{name}.hdr")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(hdr)
    return path


def test_envi_reader_registers_north_up(tmp_path):
    """ENVI's negative dy must produce a north-up transform, not a south-up one.

    from_origin() builds Affine(dx, 0, west, 0, -ysize, north), so handing it
    the raw negative dy flipped the sign and put row 0 at the bottom.
    """
    from core.io.reader import open_raster

    reader = open_raster(_write_envi_header(tmp_path))
    transform = reader.metadata.transform

    assert transform.e < 0, "row 0 must be the northern edge"
    assert transform.f == pytest.approx(4000000.0)
    # Six rows of 30 m south of the tie point.
    assert transform.f + 6 * transform.e == pytest.approx(4000000.0 - 180.0)


def test_envi_pixel_to_geo_agrees_with_the_transform(tmp_path):
    """The map-info path and the transform path must not disagree."""
    from core.io.reader import open_raster

    reader = open_raster(_write_envi_header(tmp_path))
    transform = reader.metadata.transform

    top_x, top_y = reader.pixel_to_geo(0, 0)
    bot_x, bot_y = reader.pixel_to_geo(0, 5)

    assert bot_y < top_y, "going down the image must go south"

    # Both readers report the pixel *centre*, so the map-info path has to land on
    # exactly where the affine transform puts the centre of the same pixel.
    for x, y in [(0, 0), (0, 5), (3, 2)]:
        gx, gy = reader.pixel_to_geo(x, y)
        cx, cy = transform * (x + 0.5, y + 0.5)
        assert gx == pytest.approx(cx), f"disagrees with the transform at {x},{y}"
        assert gy == pytest.approx(cy), f"disagrees with the transform at {x},{y}"


def test_envi_reader_propagates_data_ignore_value(tmp_path):
    """'data ignore value' is the ENVI nodata sentinel and must reach the metadata.

    Without it every -9999 pixel reads as signal, which corrupts statistics,
    display stretches, DOS-1 calibration and nodata-masked mosaicking.
    """
    from core.io.reader import open_raster

    plain = open_raster(_write_envi_header(tmp_path, "plain")).metadata
    assert plain.nodata is None

    flagged = open_raster(
        _write_envi_header(tmp_path, "flagged", "data ignore value = -9999\n")
    ).metadata
    assert flagged.nodata == pytest.approx(-9999.0)

    # A zero sentinel would mask every genuinely black pixel, so it is ignored.
    zeroed = open_raster(
        _write_envi_header(tmp_path, "zeroed", "data ignore value = 0\n")
    ).metadata
    assert zeroed.nodata is None


def test_synthetic_envi_writer_uses_negative_dy(tmp_path):
    """Files we write must follow the ENVI convention so they round-trip north-up."""
    import os

    from core.io.reader import open_raster
    from core.synthetic import generate_synthetic_cube, write_envi_dataset

    cube, wl, _ = generate_synthetic_cube(lines=8, samples=8, bands=3)
    base = os.path.join(tmp_path, "syn")
    hdr, _ = write_envi_dataset(base, cube, wl, interleave="bsq")

    with open(hdr, encoding="utf-8") as fh:
        content = fh.read()
    map_info = [ln for ln in content.splitlines() if ln.startswith("map info")][0]
    fields = [f.strip() for f in map_info.split("=", 1)[1].split(",")]
    assert float(fields[6]) < 0, "dy must be negative for a north-referenced image"

    reader = open_raster(hdr)
    assert reader.metadata.transform.e < 0, "our own output must read back north-up"


def test_nodata_pixels_are_excluded_from_band_statistics(tmp_path):
    """A nodata sentinel must not drag the statistics towards -9999."""
    from core.io.reader import open_raster
    from core.algorithms.statistics import calculate_band_statistics

    reader = open_raster(
        _write_envi_header(tmp_path, "stats", "data ignore value = -9999\n")
    )
    band = reader.read_band(0).astype(np.float64)
    band[:, -1] = -9999.0

    stats = calculate_band_statistics(band, nodata=reader.metadata.nodata)

    assert stats["min"] > -100.0, "the sentinel leaked into the statistics"
    # The masked column held 4, 9, 14, 19, 24 and 29; the survivors top out at 28.
    assert stats["max"] == pytest.approx(28.0)


def test_single_band_pca_mnf_and_sff_do_not_crash():
    """np.cov returns a 0-d array for one band, which eigh then rejects."""
    from core.algorithms.spectral import compute_mnf, compute_pca, spectral_feature_fitting

    cube = np.random.default_rng(0).random((1, 4, 5)).astype(np.float32)

    scores, _, _ = compute_pca(cube, num_components=1)
    assert scores.shape == (1, 4, 5)
    assert np.all(np.isfinite(scores))

    mnf, noise = compute_mnf(cube, num_components=1)
    assert mnf.shape == (1, 4, 5)
    assert noise.shape == (1,)

    ends, _ = spectral_feature_fitting(cube, num_features=0)
    assert ends.shape[1] == 1


def test_mosaic_survives_a_reader_with_no_band_details(tmp_path):
    """The band-details fallback must construct a BandInfo that actually exists."""
    from core.io.reader import open_raster
    from core.algorithms.mosaic import mosaic_rasters

    reader = open_raster(_write_envi_header(tmp_path, "nobands"))
    reader.metadata.band_details = None

    cube, meta = mosaic_rasters([reader], resample_method="nearest")

    assert cube.shape == (6, 5, 2)
    assert len(meta.band_details) == 2
    assert meta.band_details[0].name == "Band 1"


class _FakeLandsatReader:
    """Minimal reader stand-in carrying Landsat MTL rescaling metadata."""

    def __init__(self, band_name, mtl_band):
        from core.models import BandInfo, RasterMetadata

        self.metadata = RasterMetadata(
            width=2, height=2, bands=1, dtype="float32",
            crs="EPSG:32650", nodata=None,
            band_details=[
                BandInfo(index=0, name=band_name, wavelength=500.0, mtl_band=mtl_band)
            ],
            raw_header={
                "mtl_data": {
                    "L1_METADATA_FILE": {
                        "RADIOMETRIC_RESCALING": {
                            f"REFLECTANCE_MULT_BAND_{n}": n * 0.00001
                            for n in range(1, 12)
                        } | {
                            f"REFLECTANCE_ADD_BAND_{n}": -0.01 * n
                            for n in range(1, 12)
                        },
                        "IMAGE_ATTRIBUTES": {"SUN_ELEVATION": 45.0},
                    }
                }
            },
        )

    def read_band(self, index):
        return np.full((2, 2), 20000.0, dtype=np.float32)


def test_l57_bands_read_their_own_mtl_coefficients():
    """Landsat 4/5/7 number the same physical bands differently from 8/9.

    The L8/9 name table maps NIR to B5, SWIR 1 to B6 and leaves a bare
    "Thermal Infrared" to fall through on the substring "red", so on L5/7 every
    band but SWIR 2 read another band's REFLECTANCE_MULT_BAND_n.
    """
    from core.algorithms.radiometry import extract_landsat_cal_params

    # band name -> the MTL band number that actually holds its coefficients
    l57 = {
        "Blue": 1, "Green": 2, "Red": 3,
        "Near Infrared (NIR)": 4, "Shortwave Infrared 1 (SWIR 1)": 5,
        "Shortwave Infrared 2 (SWIR 2)": 7, "Thermal Infrared": 6,
    }
    for name, mtl_band in l57.items():
        params = extract_landsat_cal_params(_FakeLandsatReader(name, mtl_band))
        assert params["reflectance_mult"][0] == pytest.approx(mtl_band * 0.00001), (
            f"{name} read the wrong MTL band number"
        )
        assert params["reflectance_add"][0] == pytest.approx(-0.01 * mtl_band)


def test_band_name_fallback_still_handles_a_bare_thermal_band():
    """Without an MTL number, "Thermal Infrared" must not match the "red" rule."""
    from core.algorithms.radiometry import extract_landsat_cal_params

    reader = _FakeLandsatReader("Thermal Infrared", mtl_band=None)
    params = extract_landsat_cal_params(reader)

    # L8/9 TIRS 1 is B10; the failure mode was B4, i.e. the Red coefficient.
    assert params["reflectance_mult"][0] == pytest.approx(10 * 0.00001)


def test_dos1_is_reproducible_across_calls():
    """DOS-1 must return the same result for the same image every time.

    It samples pixels above 100k through the unseeded global RNG, so the dark
    object value used to change between runs on an unchanged scene.
    """
    from core.algorithms.radiometry import calibrate_band_to_reflectance

    rng = np.random.default_rng(1)
    band = rng.uniform(500, 30000, size=(500, 500)).astype(np.float32)

    runs = [
        calibrate_band_to_reflectance(
            band, mult=2e-5, add=-0.1, sun_elevation_deg=45.0, apply_dos=True
        )
        for _ in range(4)
    ]
    for other in runs[1:]:
        np.testing.assert_array_equal(runs[0], other)


def test_band_math_integer_casts_keep_nodata_as_nan():
    """A float-to-int cast turns NaN into INT_MIN, not into a nodata pixel."""
    from core.algorithms.indices import evaluate_band_math

    band = np.array([[np.nan, 5.0], [7.0, 9.0]])

    for expr in ("fix(band)", "int(band)", "long(band)", "byte(band)", "uint(band)"):
        out = np.ravel(np.asarray(evaluate_band_math(expr, {"band": band})))
        assert np.isnan(out[0]), f"{expr} lost the nodata pixel"
        np.testing.assert_array_equal(out[1:], np.array([5.0, 7.0, 9.0]))

    # A nodata pixel must not become a confident comparison either.
    mask = np.asarray(evaluate_band_math("band > 3", {"band": band}))
    assert mask[0, 0] == 0 or np.isnan(mask[0, 0])


def test_band_math_integer_casts_still_truncate_and_clip():
    """Preserving NaN must not change the truncation or saturation behaviour."""
    from core.algorithms.indices import evaluate_band_math

    filler = np.zeros((2, 2), dtype=np.float32)
    val = lambda expr: float(np.asarray(evaluate_band_math(expr, {"band": filler})).ravel()[0])

    assert val("fix(2.7)") == 2.0
    assert val("fix(-2.7)") == -2.0
    assert val("byte(300)") == 255.0
    assert val("byte(-5)") == 0.0
    assert val("uint(70000)") == 65535.0
    assert val("uint(-1)") == 0.0


def test_squared_distances_survive_float32_dn_magnitudes():
    """Nearest-centroid labelling must not depend on float32 rounding.

    On raw DN the squared terms are ~1e9 while the gap between competing
    classes is ~1e2, which is below float32 precision at that magnitude.
    """
    from core.algorithms.classification import _squared_distances

    rng = np.random.default_rng(0)
    data = rng.uniform(29000, 31000, size=(2000, 4)).astype(np.float32)
    centers = np.array(
        [[30000.0] * 4, [30001.0] * 4], dtype=np.float32
    )

    got = _squared_distances(data, centers)

    # Reference: an exact, cancellation-free computation in Python floats.
    ref = np.array(
        [[float(np.sum((row.astype(np.float64) - c.astype(np.float64)) ** 2)) for c in centers]
         for row in data]
    )
    np.testing.assert_allclose(got, ref, rtol=1e-9)
    np.testing.assert_array_equal(np.argmin(got, axis=1), np.argmin(ref, axis=1))


# ---------------------------------------------------------------------------
# Regression tests for the second defect-audit round.
# ---------------------------------------------------------------------------

import inspect


def test_kmeans_leaves_nodata_unclassified_instead_of_collapsing():
    """One NaN pixel used to poison a cluster centre and swallow the scene.

    mean() over a set containing NaN is NaN, every distance against that centre
    is NaN, and np.argmin then returns 0 for all of them -- so the whole image
    came back as a single class.
    """
    cube = np.full((2, 6, 6), 10.0, dtype=np.float32)
    cube[0, 0, 0] = np.nan

    class_map, centers = kmeans_clustering(cube, num_classes=2, max_iter=5)

    assert np.all(np.isfinite(centers)), "a NoData pixel must not reach the centres"
    assert class_map[0, 0] == -1, "the NoData pixel itself stays unclassified"
    assert not np.any(class_map[1:, 1:] == -1), "valid pixels must all be classified"


def test_kmeans_without_nodata_is_unchanged_by_the_mask():
    """The finite-pixel mask must not alter a result when nothing is masked."""
    rng = np.random.default_rng(7)
    cube = rng.uniform(0, 100, size=(4, 20, 20)).astype(np.float32)

    class_map, centers = kmeans_clustering(cube, num_classes=3, max_iter=10, seed=3)

    assert class_map.shape == (20, 20)
    assert class_map.min() >= 0, "no pixel is unclassified when nothing is masked"
    assert centers.shape == (3, 4)
    assert np.all(np.isfinite(centers))


def test_kmeans_tolerates_more_classes_than_pixels():
    """rng.choice(..., replace=False) raised when K exceeded the pixel count."""
    class_map, centers = kmeans_clustering(
        np.zeros((2, 1, 2), dtype=np.float32), num_classes=4, max_iter=2
    )
    assert class_map.shape == (1, 2)
    assert centers.shape[0] <= 4


def test_kmeans_on_an_entirely_masked_cube_is_all_unclassified():
    class_map, centers = kmeans_clustering(
        np.full((2, 3, 3), np.nan, dtype=np.float32), num_classes=2
    )
    assert np.all(class_map == -1)
    assert np.all(np.isfinite(centers))


def test_isodata_does_not_propagate_nodata_into_centres():
    grad = np.linspace(0.0, 1.0, 400, dtype=np.float32).reshape(20, 20)
    cube = np.stack([grad, grad * 2.0, grad * 3.0])
    cube[:, 0, 0] = np.nan

    class_map, centers = isodata_clustering(cube, initial_classes=2, max_iter=3)

    assert np.all(np.isfinite(centers))
    assert class_map[0, 0] == -1
    assert set(np.unique(class_map)) <= {-1, 0, 1, 2, 3}


def test_resample_lands_on_the_target_grid_rather_than_half_a_pixel_off():
    """grid_mode=False shifted every resampled band by half an input pixel.

    An impulse upsampled 2x lands between the output pixels under grid_mode=False
    and squarely on them under grid_mode=True.
    """
    from core.algorithms.pansharpen import resample_band_to_grid

    impulse = np.zeros((9, 9), dtype=np.float32)
    impulse[4, 4] = 1.0
    out = resample_band_to_grid(impulse, 18, 18, order=1)

    # Bilinear upsampling conserves total energy: a 2x zoom multiplies the input
    # mass by 4. Under grid_mode=False the mass came out at 4.484.
    assert out.shape == (18, 18)
    assert out.sum() == pytest.approx(4.0, rel=1e-5)
    # The impulse must land on the output pixels, not between them.
    peak = np.unravel_index(int(np.argmax(out)), out.shape)
    assert peak == (8, 8)


def test_resample_does_not_overshoot_past_the_source_range():
    """grid-constant padding let the spline overshoot below the darkest input value."""
    from core.algorithms.pansharpen import resample_band_to_grid

    block = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    out = resample_band_to_grid(block, 4, 4, order=1)

    assert out.min() >= 10.0 - 1e-4
    assert out.max() <= 40.0 + 1e-4


def test_resample_decimation_keeps_the_signal_within_range():
    from core.algorithms.pansharpen import resample_band_to_grid

    ramp = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = resample_band_to_grid(ramp, 2, 2, order=1)

    assert out.shape == (2, 2)
    # grid_mode=True samples at the target pixel centres rather than at the
    # source edges, so a decimated ramp stays ordered and inside the source
    # range instead of being pulled toward the corners.
    assert out.min() >= ramp.min()
    assert out.max() <= ramp.max()
    assert np.all(np.diff(out.ravel()) > 0), "an ascending ramp must stay ascending"


def test_envi_pixel_to_geo_returns_the_pixel_centre(tmp_path):
    """The map-info tie point names a corner; every other reader reports a centre."""
    from core.io.reader import open_raster

    reader = open_raster(_write_envi_header(tmp_path))
    transform = reader.metadata.transform

    # 30 m pixels, tie point at (1, 1) -> pixel (0, 0) centres 15 m inside.
    gx, gy = reader.pixel_to_geo(0, 0)
    assert gx == pytest.approx(500015.0)
    assert gy == pytest.approx(3999985.0)

    # A non-unit tie point must not introduce a growing error.
    import os

    data = np.arange(2 * 6 * 5, dtype=np.int16)
    data.tofile(os.path.join(tmp_path, "tie.dat"))
    with open(os.path.join(tmp_path, "tie.hdr"), "w", encoding="utf-8") as fh:
        fh.write(
            "ENVI\nsamples = 5\nlines = 6\nbands = 2\nheader offset = 0\n"
            "file type = ENVI Standard\ndata type = 12\ninterleave = bsq\n"
            "map info = {UTM, 3.000, 2.000, 500000.0, 4000000.0, 30.0, -30.0, "
            "1, North, WGS-84}\n"
        )
    tied = open_raster(os.path.join(tmp_path, "tie.hdr"))
    t = tied.metadata.transform
    for x, y in [(0, 0), (2, 3)]:
        cx, cy = t * (x + 0.5, y + 0.5)
        px, py = tied.pixel_to_geo(x, y)
        assert px == pytest.approx(cx)
        assert py == pytest.approx(cy)
    tied.close()


def test_envi_and_geotiff_readers_agree_on_pixel_geometry(tmp_path):
    """The status bar cursor reads through whichever reader is active."""
    import os

    import rasterio
    from rasterio.transform import from_origin

    from core.io.geotiff import GeoTIFFRasterReader

    envi_path = _write_envi_header(tmp_path, name="cmp")
    env_r = open_raster_helper(envi_path)

    tif = os.path.join(tmp_path, "cmp.tif")
    with rasterio.open(
        tif, "w", driver="GTiff", width=5, height=6, count=2,
        dtype="int16", crs="EPSG:32650", transform=from_origin(500000, 4000000, 30, 30),
    ) as dst:
        dst.write(np.zeros((2, 6, 5), dtype=np.int16))
    gt_r = GeoTIFFRasterReader(tif)

    for x, y in [(0, 0), (4, 5), (2, 3)]:
        assert env_r.pixel_to_geo(x, y) == pytest.approx(gt_r.pixel_to_geo(x, y))

    env_r.close()
    gt_r.close()


def open_raster_helper(path):
    from core.io.reader import open_raster

    return open_raster(path)


def test_toa_reflectance_is_not_divided_by_the_sun_angle_twice():
    """The MTL coefficients already contain 1/sin(sun_elevation).

    Dividing again inflated every reflectance by ~1/sin(theta): on this scene
    (sun at 39.7 degrees) a true 0.020 came back as 0.031.
    """
    from core.algorithms.radiometry import calibrate_band_to_reflectance

    dn = np.full((4, 4), 10000.0, dtype=np.float32)
    got = calibrate_band_to_reflectance(dn, mult=2e-5, add=-0.1, sun_elevation_deg=39.71467069)

    np.testing.assert_allclose(got, 0.1, rtol=1e-5)


def test_toa_reflectance_is_independent_of_the_supplied_sun_angle():
    from core.algorithms.radiometry import calibrate_band_to_reflectance

    dn = np.full((4, 4), 12000.0, dtype=np.float32)
    ref = calibrate_band_to_reflectance(dn, mult=2e-5, add=-0.1, sun_elevation_deg=20.0)
    for elevation in (40.0, 65.0, 80.0):
        other = calibrate_band_to_reflectance(dn, mult=2e-5, add=-0.1, sun_elevation_deg=elevation)
        np.testing.assert_allclose(other, ref, rtol=1e-6)


def test_big_endian_envi_exports_to_geotiff(tmp_path):
    """A 'byte order = 1' file advertised '>u2', which rasterio rejects outright."""
    import os

    import rasterio

    from core.io.envi import ENVIRasterReader
    from core.io.writer import export_raster

    os.makedirs(tmp_path, exist_ok=True)
    with open(os.path.join(tmp_path, "be.hdr"), "w", encoding="utf-8") as fh:
        fh.write(
            "ENVI\nsamples = 4\nlines = 3\nbands = 1\ndata type = 12\n"
            "header offset = 0\nfile type = ENVI Standard\nbyte order = 1\n"
        )
    source = (np.arange(12, dtype=np.int16) * 1000).astype(">i2")
    source.tofile(os.path.join(tmp_path, "be.dat"))

    reader = ENVIRasterReader(os.path.join(tmp_path, "be.hdr"))
    # The advertised dtype carries no byte order; the memmap still reads big-endian.
    assert reader.metadata.dtype == "uint16"
    np.testing.assert_array_equal(reader.read_band(0).ravel(), source.astype(np.int16).ravel())

    out = os.path.join(tmp_path, "be_out.tif")
    export_raster(reader, out)
    reader.close()

    with rasterio.open(out) as ds:
        assert ds.dtypes == ("uint16",)
        np.testing.assert_array_equal(ds.read(1).ravel(), source.astype(np.int16).ravel())


def test_subset_keeps_per_band_metadata_aligned_when_the_list_is_short(tmp_path):
    """A band-names list shorter than the band count used to shift every name."""
    import os

    from core.algorithms.subset import resize_subset_raster
    from core.io.reader import open_raster

    lines, samples, bands = 8, 8, 3
    data = np.arange(bands * lines * samples, dtype=np.int16).reshape(bands, lines, samples)
    data.tofile(os.path.join(tmp_path, "s.dat"))
    with open(os.path.join(tmp_path, "s.hdr"), "w", encoding="utf-8") as fh:
        fh.write(
            "ENVI\nsamples = 8\nlines = 8\nbands = 3\ndata type = 12\nheader offset = 0\n"
            "file type = ENVI Standard\ninterleave = bsq\n"
            "band names = {Coastal}\n"          # deliberately short
            "map info = {UTM, 1.000, 1.000, 500000.0, 4000000.0, 30.0, -30.0, "
            "1, North, WGS-84}\n"
        )
    reader = open_raster(os.path.join(tmp_path, "s.hdr"))

    _, meta = resize_subset_raster(reader, 0, 8, 0, 8, selected_bands=[0, 2])

    names = meta.raw_header["band names"]
    assert len(names) == 2, "the list must stay aligned with the band count"
    assert names[0] == "Coastal"
    assert names[1] == "Band 3"

    reader.close()


def test_memory_reader_keeps_a_small_bhw_cube_intact():
    """'last axis <= 100 means bands' turned an 8-band 4x5 chip into 5 bands."""
    from core.io.memory import MemoryRasterReader

    cube = np.arange(8 * 4 * 5, dtype=np.float32).reshape(8, 4, 5)
    reader = MemoryRasterReader(cube, name="chip")

    assert reader.metadata.bands == 8
    assert reader.metadata.lines == 4
    assert reader.metadata.samples == 5
    np.testing.assert_array_equal(reader.read_band(7).ravel(), cube[7].ravel())


def test_memory_reader_honours_an_explicit_layout():
    from core.io.memory import MemoryRasterReader

    hwb = np.zeros((4, 5, 3), dtype=np.float32)
    assert MemoryRasterReader(hwb, layout="hwb").metadata.bands == 3

    bhw = np.zeros((3, 4, 5), dtype=np.float32)
    reader = MemoryRasterReader(bhw, layout="bhw")
    assert (reader.metadata.bands, reader.metadata.lines, reader.metadata.samples) == (3, 4, 5)

    with pytest.raises(ValueError):
        MemoryRasterReader(bhw, layout="nonsense")


def test_stacking_passes_src_nodata_so_the_scene_edge_has_no_dark_halo(tmp_path):
    """Without src_nodata the warp blends the NoData sentinel into valid pixels."""
    from core.algorithms.stacking import stack_bands
    from core.io.memory import MemoryRasterReader
    from core.models import BandInfo, RasterMetadata

    meta_a = RasterMetadata(
        width=8, height=8, bands=1, dtype="uint16", nodata=0, transform=_utm_affine(),
        crs="EPSG:32650", band_details=[BandInfo(index=0, name="A", wavelength=None)],
    )
    meta_b = RasterMetadata(
        width=8, height=8, bands=1, dtype="uint16", nodata=0, transform=_utm_affine(dx=240.0),
        crs="EPSG:32650", band_details=[BandInfo(index=0, name="B", wavelength=None)],
    )

    reader_a = MemoryRasterReader(
        np.full((8, 8), 1000, dtype=np.float32), name="a", parent_metadata=meta_a
    )
    reader_b = MemoryRasterReader(
        np.full((8, 8), 2000, dtype=np.float32), name="b", parent_metadata=meta_b
    )

    cube, _ = stack_bands([(reader_a, 0), (reader_b, 0)], resampling_method="bilinear")

    # Band A occupies the left half of the reference grid; its values must not be
    # dragged toward the 0 sentinel coming from beyond its own footprint.
    covered = cube[:8, :8, 0] > 0
    assert covered.any()
    np.testing.assert_allclose(cube[:8, :8, 0][covered], 1000.0, rtol=1e-3)


def _utm_affine(dx=0.0):
    from rasterio.transform import from_origin

    return from_origin(500000.0 + dx, 4000000.0, 30.0, 30.0)


def test_constant_band_histogram_peak_sits_where_the_value_is():
    """A constant band filed its counts under bin 0 while the value sat in bin 128."""
    from core.algorithms.statistics import calculate_band_statistics

    stats = calculate_band_statistics(np.full((8, 8), 100.0), bins=256)
    counts, edges = stats["hist_counts"], stats["bin_edges"]

    assert counts.sum() == 64
    peak = int(np.argmax(counts))
    assert edges[peak] <= 100.0 < edges[peak + 1], "the peak bin must contain the value"
    assert peak == 128


def test_constant_band_histogram_matches_across_bin_counts():
    from core.algorithms.statistics import calculate_band_statistics

    for bins in (10, 64, 256):
        stats = calculate_band_statistics(np.full((8, 8), 42.0), bins=bins)
        counts, edges = stats["hist_counts"], stats["bin_edges"]
        peak = int(np.argmax(counts))
        assert counts[peak] == 64
        assert edges[peak] <= 42.0 < edges[peak + 1]


def test_roi_mean_spectrum_marks_a_fully_masked_band_as_nan():
    """0.0 is indistinguishable from a genuinely black mean spectrum."""
    from core.io.memory import MemoryRasterReader
    from core.models import BandInfo, RasterMetadata
    from core.roi import ROI

    meta = RasterMetadata(
        width=8, height=8, bands=2, dtype="uint16", nodata=0,
        band_details=[BandInfo(index=i, name=f"B{i + 1}", wavelength=None) for i in range(2)],
    )
    data = np.full((2, 8, 8), 500.0, dtype=np.float32)
    data[1, :, :] = 0.0  # the whole second band is NoData
    reader = MemoryRasterReader(data, name="roi_t", parent_metadata=meta)

    spectrum = ROI(roi_id="r", name="roi", bbox=(0, 0, 4, 4)).calculate_mean_spectrum(reader)

    assert spectrum[0] == pytest.approx(500.0)
    assert np.isnan(spectrum[1]), "an all-NoData band must be NaN, not a fake 0.0"


def test_mlc_chi_square_fallback_distinguishes_each_alpha():
    """Every alpha <= 0.05 used to collapse onto the same one-sided quantile.

    Collapsing them made the 1% rejection far too permissive, so pixels that
    should have stayed unclassified got assigned to the nearest class.
    """
    import core.algorithms.classification as mod

    src = inspect.getsource(mod.maximum_likelihood_classification)
    # The one-sided normal quantiles must all be present and distinct.
    for quantile in ("2.3263", "1.6449", "1.2816"):
        assert quantile in src, f"missing the alpha={quantile} quantile"
    assert "1.6449 if probability_threshold <= 0.05" not in src, "alphas are still merged"


def test_mlc_runs_for_each_tabulated_alpha_with_and_without_scipy():
    from core.algorithms.classification import maximum_likelihood_classification

    bands = 3
    rng = np.random.default_rng(11)
    cube = rng.normal(0, 1, size=(bands, 30, 30)).astype(np.float32)
    training = {
        0: rng.normal(-20, 1, size=(40, bands)).astype(np.float32),
        1: rng.normal(20, 1, size=(40, bands)).astype(np.float32),
    }

    for alpha in (0.01, 0.05, 0.10):
        class_map, _ = maximum_likelihood_classification(
            cube, training, probability_threshold=alpha
        )
        assert class_map.dtype == np.int32
        assert set(np.unique(class_map)) <= {-1, 0, 1}


def test_mlc_unclassified_fraction_grows_as_alpha_tightens():
    """With no scipy the fallback must still tighten with alpha.

    Wide, overlapping training clouds put borderline pixels between the 1%, 5%
    and 10% chi-square thresholds, so the three alphas give different answers.
    """
    import builtins

    from core.algorithms.classification import maximum_likelihood_classification

    bands = 4
    rng = np.random.default_rng(5)
    cube = rng.normal(0, 7, size=(bands, 25, 25)).astype(np.float32)
    training = {
        0: rng.normal(-6, 9, size=(300, bands)).astype(np.float32),
        1: rng.normal(6, 9, size=(300, bands)).astype(np.float32),
    }

    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("scipy disabled for the test")
        return real_import(name, *args, **kwargs)

    rejected = {}
    for alpha in (0.01, 0.05, 0.10):
        builtins.__import__ = no_scipy
        try:
            class_map, _ = maximum_likelihood_classification(
                cube, training, probability_threshold=alpha
            )
        finally:
            builtins.__import__ = real_import
        rejected[alpha] = int(np.count_nonzero(class_map == -1))

    assert rejected[0.01] < rejected[0.05] < rejected[0.10]


def test_landsat_mtl_parse_failure_does_not_leak_the_reference_dataset(tmp_path):
    """A non-numeric SUN_ELEVATION raised after the GDAL handle was already owned."""
    import os

    from core.io.landsat import LandsatMTLReader

    scene = tmp_path / "scene"
    os.makedirs(scene, exist_ok=True)
    import rasterio
    from rasterio.transform import from_origin

    tif = scene / "B1.TIF"
    with rasterio.open(
        tif, "w", driver="GTiff", width=4, height=4, count=1, dtype="uint16",
        crs="EPSG:32650", transform=from_origin(500000, 4000000, 30, 30),
    ) as dst:
        dst.write(np.ones((1, 4, 4), dtype=np.uint16) * 100)

    (scene / "MTL.txt").write_text(
        "GROUP = L1_METADATA_FILE\n"
        "  GROUP = PRODUCT_METADATA\n"
        "    PROCESSING_LEVEL = L1TP\n"
        "    PRODUCT_TYPE = L8\n"
        "    SPACECRAFT_ID = LANDSAT_8\n"
        "    SENSOR_ID = OLI_TIRS\n"
        "    DATE_ACQUIRED = 2021-01-19\n"
        "    FILE_NAME_BAND_1 = B1.TIF\n"
        "  END_GROUP = PRODUCT_METADATA\n"
        "  GROUP = IMAGE_ATTRIBUTES\n"
        "    SUN_ELEVATION = not-a-number\n"
        "  END_GROUP = IMAGE_ATTRIBUTES\n"
        "  GROUP = RADIOMETRIC_RESCALING\n"
        "    REFLECTANCE_MULT_BAND_1 = 2.0000E-05\n"
        "    REFLECTANCE_ADD_BAND_1 = -0.100000\n"
        "  END_GROUP = RADIOMETRIC_RESCALING\n"
        "END_GROUP = L1_METADATA_FILE\n",
        encoding="utf-8",
    )

    reader = LandsatMTLReader(str(scene / "MTL.txt"))

    # The malformed field falls back rather than propagating out of __init__,
    # and the file is released so the directory can be removed on Windows.
    assert reader.metadata.raw_header["sun_elevation"] == 0.0
    assert reader.metadata.raw_header["earth_sun_distance"] == 1.0
    reader.close()
