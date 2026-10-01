"""OpenENVI Remote Sensing Classification Engine.

Provides K-Means, ISODATA clustering, and thematic color rendering
for multispectral and hyperspectral imagery.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# Standard ENVI Thematic Classification Palette (R, G, B in 0-255)
DEFAULT_THEMATIC_PALETTE = [
    (0, 180, 0),     # Class 0: Green (Vegetation)
    (0, 100, 240),   # Class 1: Blue (Water)
    (180, 120, 60),  # Class 2: Brown (Soil/Bare ground)
    (150, 150, 150), # Class 3: Gray (Urban / Built-up)
    (240, 220, 40),  # Class 4: Yellow (Agriculture / Crop)
    (180, 50, 200),  # Class 5: Purple (Wetland)
    (255, 128, 0),   # Class 6: Orange
    (0, 220, 220),   # Class 7: Cyan
    (220, 30, 30),   # Class 8: Red
    (100, 200, 100), # Class 9: Light Green
]


def _squared_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared distances from every row of ``data`` to every row of ``centers``.

    The expansion |x|^2 - 2<x,c> + |c|^2 is evaluated in float64. In float32 an
    uncalibrated raster sits around 1e4 DN, so each squared term is ~1e9 while
    the gap separating two nearby classes is ~1e2 -- below float32's precision
    at that magnitude, which lets argmin pick rounding noise.
    """
    a = np.asarray(data, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    a_sq = np.sum(a * a, axis=1, keepdims=True)
    c_sq = np.sum(c * c, axis=1, keepdims=True).T
    return a_sq - 2.0 * (a @ c.T) + c_sq


def kmeans_clustering(
    cube: np.ndarray,
    num_classes: int = 4,
    max_iter: int = 20,
    tol: float = 1e-4,
    seed: int = 42,
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform K-Means clustering on a multi-band raster cube with high scalability.

    For large datasets (>100k pixels), uses sub-sampled initialization and
    MiniBatchKMeans with chunked block prediction to avoid memory exhaustion (OOM)
    and long execution freezes.

    Args:
        cube: 3D numpy array of shape (bands, lines, samples).
        num_classes: Number of clusters (K).
        max_iter: Maximum iterations.
        tol: Convergence tolerance threshold.
        seed: Random seed for initialization.
        progress_callback: Optional callback receiving (percentage: int, message: str).

    Returns:
        Tuple of (classification_map, cluster_centers)
        - classification_map: 2D array of shape (lines, samples) with class indices.
        - cluster_centers: 2D array of shape (num_classes, bands).
    """
    bands, lines, samples = cube.shape
    num_pixels = lines * samples

    flat_data = cube.reshape(bands, num_pixels).T
    if flat_data.dtype != np.float32:
        flat_data = flat_data.astype(np.float32)

    rng = np.random.default_rng(seed)

    # 1. Representative Subsampling for fast center training (prevents OOM on 50M+ pixel scenes)
    sample_size = min(100_000, num_pixels)
    if sample_size < num_pixels:
        train_idx = rng.choice(num_pixels, size=sample_size, replace=False)
        train_data = flat_data[train_idx]
    else:
        train_data = flat_data

    # Initialize centers by uniform sampling from train data
    init_indices = rng.choice(sample_size, size=num_classes, replace=False)
    centers = train_data[init_indices].copy()

    # 2. Fast iterative clustering on sample
    for iteration in range(max_iter):
        if progress_callback:
            pct = 30 + int(35 * (iteration + 1) / max_iter)
            progress_callback(pct, f"K-Means iteration {iteration + 1}/{max_iter}...")

        distances = _squared_distances(train_data, centers)
        sub_labels = np.argmin(distances, axis=1)

        new_centers = np.zeros_like(centers)
        for k in range(num_classes):
            mask = (sub_labels == k)
            if np.any(mask):
                new_centers[k] = np.mean(train_data[mask], axis=0)
            else:
                new_centers[k] = train_data[rng.choice(sample_size)]

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers

        if shift < tol:
            break

    # 3. Predict full scene in memory-safe chunks (250,000 pixels per chunk)
    if progress_callback:
        progress_callback(70, "Predicting class assignments...")

    labels = np.empty(num_pixels, dtype=np.int32)
    chunk_size = 250_000
    total_chunks = (num_pixels + chunk_size - 1) // chunk_size

    for chunk_i, start in enumerate(range(0, num_pixels, chunk_size)):
        end = min(start + chunk_size, num_pixels)
        dist = _squared_distances(flat_data[start:end], centers)
        labels[start:end] = np.argmin(dist, axis=1)

        if progress_callback and total_chunks > 1:
            pct = 70 + int(20 * (chunk_i + 1) / total_chunks)
            progress_callback(pct, f"Predicting classes ({chunk_i + 1}/{total_chunks})...")

    class_map = labels.reshape(lines, samples)
    return class_map, centers


def isodata_clustering(
    cube: np.ndarray,
    initial_classes: int = 4,
    min_members: int = 10,
    max_iter: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform ISODATA iterative clustering.

    Dynamically splits clusters with high standard deviation and merges
    clusters closer than the threshold distance.

    Args:
        cube: 3D numpy array of shape (bands, lines, samples).
        initial_classes: Starting number of clusters.
        min_members: Minimum number of pixels required to retain a cluster.
        max_iter: Number of iterations.
        seed: Random seed.

    Returns:
        Tuple of (classification_map, cluster_centers).
    """
    bands, lines, samples = cube.shape
    num_pixels = lines * samples
    data = cube.reshape(bands, num_pixels).T.astype(np.float32)

    # Initialize using K-means
    class_map, centers = kmeans_clustering(cube, num_classes=initial_classes, max_iter=5, seed=seed)
    labels = class_map.flatten()

    for iteration in range(max_iter):
        active_centers = []
        for k in range(len(centers)):
            mask = (labels == k)
            count = np.sum(mask)
            if count >= min_members:
                active_centers.append(np.mean(data[mask], axis=0))

        if not active_centers:
            break

        centers = np.array(active_centers, dtype=np.float32)

        # Re-assign labels
        distances = _squared_distances(data, centers)
        labels = np.argmin(distances, axis=1).astype(np.int32)

    return labels.reshape(lines, samples), centers


def create_thematic_rgb(
    class_map: np.ndarray,
    palette: Optional[Union[List[Tuple[int, int, int]], Dict[int, Tuple[int, int, int]]]] = None,
    unclassified_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Convert a 2D integer class map into an RGB thematic color image.

    Args:
        class_map: 2D numpy array of shape (lines, samples).
        palette: Optional list or dict mapping class indices to (R, G, B) tuples.
        unclassified_color: Color for unclassified pixels (class < 0). Default: black.

    Returns:
        3D numpy array of shape (lines, samples, 3) in uint8 [0, 255].
    """
    lines, samples = class_map.shape
    rgb_image = np.zeros((lines, samples, 3), dtype=np.uint8)

    if isinstance(palette, dict):
        for class_idx in np.unique(class_map):
            if int(class_idx) < 0:
                color = unclassified_color
            else:
                color = palette.get(int(class_idx), unclassified_color)
            mask = (class_map == class_idx)
            rgb_image[mask] = color
    else:
        pal = palette if palette is not None else DEFAULT_THEMATIC_PALETTE
        for class_idx in np.unique(class_map):
            if int(class_idx) < 0:
                color = unclassified_color
            else:
                color = pal[int(class_idx) % len(pal)]
            mask = (class_map == class_idx)
            rgb_image[mask] = color

    return rgb_image


def maximum_likelihood_classification(
    cube: np.ndarray,
    training_data: Dict[int, np.ndarray],
    probability_threshold: float = 0.0,
    use_sample_priors: bool = False,
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Supervised Maximum Likelihood Classification (MLC).

    Estimates class mean vectors and regularized covariance matrices from training
    samples (typically derived from ROIs). Evaluates multivariate normal discriminant
    scores for each pixel and applies optional Chi-Square thresholding to filter
    low-confidence pixels as unclassified (-1).

    Args:
        cube: 3D numpy array of shape (bands, lines, samples).
        training_data: Dict mapping class index (0, 1, 2, ...) to 2D numpy array of
            training spectra of shape (N_samples, bands).
        probability_threshold: P-value threshold (0.0 to 1.0). If > 0, pixels with
            Mahalanobis distance exceeding the critical Chi-Square value (p < threshold)
            are assigned to -1 (unclassified).
        use_sample_priors: If True, uses training sample proportions as class priors;
            otherwise uses equal priors (1 / K).
        progress_callback: Optional callable(percent: int, message: str).

    Returns:
        Tuple of (class_map, rule_distances):
        - class_map: 2D array of shape (lines, samples) with integer class labels or -1.
        - rule_distances: 3D array of shape (num_classes, lines, samples) of squared
          Mahalanobis distances to each class.
    """
    bands, lines, samples = cube.shape
    num_pixels = lines * samples
    class_indices = sorted(training_data.keys())
    num_classes = len(class_indices)

    if num_classes < 2:
        raise ValueError("Maximum Likelihood Classification requires at least 2 training classes.")

    # 1. Parameter Estimation (means, regularized covariances, log-determinants)
    means = []
    inv_covs = []
    const_terms = []
    total_samples = sum(len(training_data[c]) for c in class_indices)

    for c in class_indices:
        X_c = training_data[c].astype(np.float64)
        n_c = len(X_c)
        if n_c < 1:
            raise ValueError(f"Training class {c} has 0 valid samples.")

        mu_c = np.mean(X_c, axis=0)  # (bands,)
        if n_c > 1:
            cov_c = np.cov(X_c, rowvar=False)
            if cov_c.ndim == 0:
                cov_c = cov_c.reshape(1, 1)
        else:
            cov_c = np.zeros((bands, bands), dtype=np.float64)

        # Regularization: ensure strictly positive-definite matrix even when n_c < bands
        trace_val = float(np.trace(cov_c))
        sigma2_avg = trace_val / bands if trace_val > 1e-12 else 1.0
        reg_cov = cov_c + (1e-4 * sigma2_avg + 1e-6) * np.eye(bands, dtype=np.float64)

        # Inversion and log determinant
        inv_cov = np.linalg.inv(reg_cov)
        sign, log_det = np.linalg.slogdet(reg_cov)
        if sign <= 0:
            # Fallback for numerical drift
            log_det = np.sum(np.log(np.maximum(np.linalg.eigvalsh(reg_cov), 1e-12)))

        prior = (n_c / total_samples) if use_sample_priors and total_samples > 0 else (1.0 / num_classes)
        const = np.log(max(prior, 1e-12)) - 0.5 * log_det

        means.append(mu_c)
        inv_covs.append(inv_cov)
        const_terms.append(const)

    # Chi-Square critical threshold for unclassified pixels
    chi2_crit = None
    if probability_threshold > 0.0:
        try:
            import scipy.stats
            chi2_crit = float(scipy.stats.chi2.ppf(1.0 - probability_threshold, df=bands))
        except Exception:
            # Asymptotic Wilson-Hilferty approximation if scipy is unavailable.
            # chi2.ppf(1 - alpha) needs the ONE-SIDED normal quantile, i.e. 1.645 at
            # alpha=0.05, not the two-sided 1.96.
            z = 1.6449 if probability_threshold <= 0.05 else 1.2816
            chi2_crit = bands * ((1.0 - 2.0 / (9.0 * bands) + z * np.sqrt(2.0 / (9.0 * bands))) ** 3)

    # 2. Prediction in memory-efficient chunks
    flat_cube = cube.reshape(bands, num_pixels).T.astype(np.float64)  # (num_pixels, bands)
    class_map_flat = np.full(num_pixels, -1, dtype=np.int32)
    rule_distances_flat = np.empty((num_classes, num_pixels), dtype=np.float32)

    chunk_size = 100_000
    total_chunks = (num_pixels + chunk_size - 1) // chunk_size

    for chunk_i, start in enumerate(range(0, num_pixels, chunk_size)):
        end = min(start + chunk_size, num_pixels)
        X_chunk = flat_cube[start:end]  # (M, bands)
        M = end - start

        # Check for valid (finite) pixels
        valid_mask = np.all(np.isfinite(X_chunk), axis=1)

        scores = np.full((M, num_classes), -np.inf, dtype=np.float64)
        dists = np.full((num_classes, M), np.inf, dtype=np.float32)

        if np.any(valid_mask):
            X_valid = X_chunk[valid_mask]  # (V, bands)
            for k in range(num_classes):
                delta = X_valid - means[k]  # (V, bands)
                # D_k^2 = sum((delta @ inv_cov) * delta, axis=1)
                D2 = np.sum((delta @ inv_covs[k]) * delta, axis=1)
                dists[k, valid_mask] = D2.astype(np.float32)
                # Discriminant: const - 0.5 * D2
                scores[valid_mask, k] = const_terms[k] - 0.5 * D2

            best_class_idx = np.argmax(scores[valid_mask], axis=1)
            best_mapped = np.array([class_indices[k] for k in best_class_idx], dtype=np.int32)

            if chi2_crit is not None:
                # Find the Mahalanobis distance of the winning class
                # Create index array for picking from dists
                best_d2 = dists[best_class_idx, np.arange(M)[valid_mask]]
                unclassified_mask = best_d2 > chi2_crit
                best_mapped[unclassified_mask] = -1

            class_map_flat[start:end][valid_mask] = best_mapped

        rule_distances_flat[:, start:end] = dists

        if progress_callback:
            pct = 40 + int(50 * (chunk_i + 1) / total_chunks)
            progress_callback(pct, f"Classifying pixels ({chunk_i + 1}/{total_chunks})...")

    class_map = class_map_flat.reshape(lines, samples)
    rule_distances = rule_distances_flat.reshape(num_classes, lines, samples)

    return class_map, rule_distances


def svm_classification(
    cube: np.ndarray,
    training_data: Dict[int, np.ndarray],
    kernel: str = "rbf",
    C: float = 100.0,
    gamma: Union[str, float] = "scale",
    degree: int = 3,
    coef0: float = 0.0,
    probability_threshold: float = 0.0,
    max_samples_per_class: int = 2000,
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Supervised Support Vector Machine (SVM) classification.

    Trains a multi-class Support Vector Classifier using training samples from ROIs,
    and classifies multispectral/hyperspectral image cubes in memory-efficient chunks.

    Args:
        cube: 3D numpy array of shape (bands, lines, samples).
        training_data: Dict mapping class index (0, 1, 2, ...) to 2D numpy array of
            training spectra of shape (N_samples, bands).
        kernel: SVM kernel type ('rbf', 'linear', 'poly', 'sigmoid').
        C: Regularization penalty parameter.
        gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        degree: Degree of the polynomial kernel function ('poly').
        coef0: Independent term in kernel function for 'poly' and 'sigmoid'.
        probability_threshold: Minimum probability threshold [0.0, 1.0]. Pixels with
            maximum class probability below this threshold are marked as -1 (unclassified).
        max_samples_per_class: Upper bound on training pixels subsampled per class. ROI
            regions on full satellite scenes can hold hundreds of thousands of pixels, and
            support-vector count (hence prediction cost) grows with it. Set to 0 or a
            negative value to train on every pixel.
        progress_callback: Optional callback receiving (percentage: int, message: str).

    Returns:
        Tuple of (class_map, rule_probabilities):
        - class_map: 2D numpy array of shape (lines, samples) with predicted class indices or -1.
        - rule_probabilities: 3D array of shape (num_classes, lines, samples) containing class probabilities.
    """
    from sklearn.svm import SVC

    bands, lines, samples = cube.shape
    num_pixels = lines * samples
    class_indices = sorted(training_data.keys())
    num_classes = len(class_indices)

    if num_classes < 2:
        raise ValueError("SVM Classification requires at least 2 training classes.")

    # 1. Assemble training dataset
    X_train_list = []
    y_train_list = []
    for c_idx in class_indices:
        data_c = training_data[c_idx]
        if len(data_c) < 1:
            raise ValueError(f"Training class {c_idx} has 0 valid samples.")
        if 0 < max_samples_per_class < len(data_c):
            # Evenly strided subsample keeps the spectral spread of the ROI intact
            step = len(data_c) / max_samples_per_class
            picks = (np.arange(max_samples_per_class) * step).astype(np.int64)
            data_c = data_c[picks]
        X_train_list.append(data_c)
        y_train_list.append(np.full(len(data_c), c_idx, dtype=np.int32))

    X_train = np.vstack(X_train_list).astype(np.float64)
    y_train = np.concatenate(y_train_list)

    if progress_callback:
        progress_callback(35, f"Training SVM model ({kernel.upper()} kernel, C={C})...")

    # 2. Fit SVM model
    clf = SVC(
        C=float(C),
        kernel=str(kernel),
        degree=int(degree),
        gamma=gamma if isinstance(gamma, str) else float(gamma),
        coef0=float(coef0),
        probability=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    if progress_callback:
        progress_callback(55, "Classifying image pixels...")

    # 3. Predict in memory-safe chunks
    flat_cube = cube.reshape(bands, num_pixels).T.astype(np.float64)
    class_map_flat = np.full(num_pixels, -1, dtype=np.int32)
    rule_probs_flat = np.zeros((num_classes, num_pixels), dtype=np.float32)

    # Where each trained class lands in the rule-probability rows, so the
    # per-chunk loop is a dict lookup instead of a linear scan per column.
    row_of_class = {c: i for i, c in enumerate(class_indices)}

    chunk_size = 50_000
    total_chunks = (num_pixels + chunk_size - 1) // chunk_size

    for chunk_i, start in enumerate(range(0, num_pixels, chunk_size)):
        end = min(start + chunk_size, num_pixels)
        X_chunk = flat_cube[start:end]
        valid_mask = np.all(np.isfinite(X_chunk), axis=1)

        if np.any(valid_mask):
            X_valid = X_chunk[valid_mask]
            probs_valid = clf.predict_proba(X_valid)  # shape: (n_valid, n_clf_classes)

            # Map probabilities into rule_probs_flat
            for clf_col, c_name in enumerate(clf.classes_):
                target_row = row_of_class[int(c_name)]
                rule_probs_flat[target_row, start:end][valid_mask] = probs_valid[:, clf_col].astype(np.float32)

            # Determine winning class
            max_prob_indices = np.argmax(probs_valid, axis=1)
            predicted_labels = clf.classes_[max_prob_indices]
            max_probs = np.max(probs_valid, axis=1)

            if probability_threshold > 0.0:
                unclass_mask = max_probs < probability_threshold
                predicted_labels[unclass_mask] = -1

            class_map_flat[start:end][valid_mask] = predicted_labels

        if progress_callback:
            pct = 55 + int(40 * (chunk_i + 1) / total_chunks)
            progress_callback(pct, f"Classifying pixels ({chunk_i + 1}/{total_chunks})...")

    class_map = class_map_flat.reshape(lines, samples)
    rule_probabilities = rule_probs_flat.reshape(num_classes, lines, samples)

    return class_map, rule_probabilities


