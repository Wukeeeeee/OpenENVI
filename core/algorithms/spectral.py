"""OpenENVI Hyperspectral Algorithms Engine.

Provides Principal Component Analysis (PCA), Minimum Noise Fraction (MNF),
and Spectral Angle Mapper (SAM) for hyperspectral data reduction and classification.
"""

from typing import Optional, Tuple
import numpy as np


def compute_pca(
    cube: np.ndarray,
    num_components: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Principal Component Analysis (PCA) on a hyperspectral cube.

    Args:
        cube: 3D numpy array of shape (bands, lines, samples).
        num_components: Number of principal components to return.

    Returns:
        Tuple of (score_cube, eigenvalues, explained_variance_ratio)
        - score_cube: np.ndarray of shape (num_components, lines, samples)
        - eigenvalues: 1D array of eigenvalues sorted in descending order
        - explained_variance_ratio: 1D array of variance explained by each component
    """
    bands, lines, samples = cube.shape
    num_components = min(num_components, bands)
    total_pixels = lines * samples

    # Pixels containing NaN/Inf cannot contribute to the covariance estimate
    finite_mask = np.all(np.isfinite(cube), axis=0)

    # For large datasets (> 500k pixels), compute covariance on spatial subsample
    if total_pixels > 500_000:
        step = max(1, int(np.sqrt(total_pixels / 250_000)))
        sub_cube = cube[:, ::step, ::step]
        sub_finite = np.all(np.isfinite(sub_cube), axis=0)
        sub_flat = sub_cube.reshape(bands, -1).T.astype(np.float32)[sub_finite.reshape(-1)]
        mean_vec = np.mean(sub_flat, axis=0)
        cov = np.cov(sub_flat - mean_vec, rowvar=False)
    else:
        flat_data = cube.reshape(bands, -1).T.astype(np.float32)[finite_mask.reshape(-1)]
        mean_vec = np.mean(flat_data, axis=0)
        cov = np.cov(flat_data - mean_vec, rowvar=False)

    # Eigendecomposition on (bands, bands) covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = np.sum(eigenvalues)
    explained_var = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)

    # Project centered data onto top eigenvectors band-by-band to minimize peak memory
    top_eigenvectors = eigenvectors[:, :num_components]
    score_cube = np.zeros((num_components, lines, samples), dtype=np.float32)

    for k in range(num_components):
        for b in range(bands):
            coeff = float(top_eigenvectors[b, k])
            if abs(coeff) > 1e-7:
                score_cube[k] += coeff * (cube[b].astype(np.float32) - float(mean_vec[b]))

    return score_cube, eigenvalues, explained_var[:num_components]


def compute_mnf(
    cube: np.ndarray,
    num_components: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Minimum Noise Fraction (MNF) transform.

    Estimates sensor noise covariance using shift differences, whitens the noise,
    and performs principal component decomposition on the whitened signal.

    Args:
        cube: 3D numpy array of shape (bands, lines, samples).
        num_components: Number of MNF components to return.

    Returns:
        Tuple of (mnf_cube, eigenvalues)
        - mnf_cube: np.ndarray shape (num_components, lines, samples)
        - eigenvalues: MNF eigenvalues (Signal-to-Noise Ratio + 1)
    """
    bands, lines, samples = cube.shape
    num_components = min(num_components, bands)
    total_pixels = lines * samples

    # 1. Estimate noise from spatial shift differences: Delta = X[:, :, 1:] - X[:, :, :-1]
    #    Non-finite pixels are excluded from both the noise and signal estimates.
    if total_pixels > 500_000:
        step = max(1, int(np.sqrt(total_pixels / 250_000)))
        diff_h = cube[:, ::step, 1::step] - cube[:, ::step, :-1:step]
        diff_ok = np.all(np.isfinite(diff_h), axis=0)
        diff_flat = diff_h.reshape(bands, -1).T.astype(np.float32)[diff_ok.reshape(-1)]
        noise_cov = np.cov(diff_flat, rowvar=False) * 0.5

        data_sub = cube[:, ::step, ::step].reshape(bands, -1).T.astype(np.float32)
        data_ok = np.all(np.isfinite(data_sub), axis=1)
        data_sub = data_sub[data_ok]
        mean_vec = np.mean(data_sub, axis=0)
        total_cov = np.cov(data_sub - mean_vec, rowvar=False)
    else:
        diff_h = cube[:, :, 1:] - cube[:, :, :-1]
        diff_ok = np.all(np.isfinite(diff_h), axis=0)
        diff_flat = diff_h.reshape(bands, -1).T.astype(np.float32)[diff_ok.reshape(-1)]
        noise_cov = np.cov(diff_flat, rowvar=False) * 0.5

        data_flat = cube.reshape(bands, -1).T.astype(np.float32)
        data_ok = np.all(np.isfinite(data_flat), axis=1)
        data_flat = data_flat[data_ok]
        mean_vec = np.mean(data_flat, axis=0)
        total_cov = np.cov(data_flat - mean_vec, rowvar=False)

    # Regularize noise covariance for numerical stability
    noise_cov += np.eye(bands) * 1e-6

    # 2. Whiten the noise, then run principal component decomposition on the whitened
    #    signal. The Sigma_n^(-1/2) factor is essential: projecting with the eigenvectors
    #    of Sigma_n^-1 * Sigma_s alone leaves the component noise at Sigma_n instead of
    #    the identity, so the reported SNR values would not match the delivered
    #    components.
    try:
        W = np.linalg.pinv(np.linalg.cholesky(noise_cov)).T
        whitened_cov = np.dot(W, np.dot(total_cov, W.T))
        eigenvalues, whitening_evecs = np.linalg.eig(whitened_cov)
        # Take real parts
        eigenvalues = np.real(eigenvalues)
        whitening_evecs = np.real(whitening_evecs)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        whitening_evecs = whitening_evecs[:, idx]

        # MNF transform matrix: M = Sigma_n^(-1/2) * V
        top_vectors = np.dot(W, whitening_evecs[:, :num_components])
        mnf_cube = np.zeros((num_components, lines, samples), dtype=np.float32)

        for k in range(num_components):
            for b in range(bands):
                coeff = float(top_vectors[b, k])
                if abs(coeff) > 1e-7:
                    mnf_cube[k] += coeff * (cube[b].astype(np.float32) - float(mean_vec[b]))

        return mnf_cube, eigenvalues[:num_components]
    except Exception:
        # Fallback to standard PCA if the noise covariance is not positive definite
        score_cube, eig, _ = compute_pca(cube, num_components=num_components)
        return score_cube, eig[:num_components]


def spectral_angle_mapper(
    cube: np.ndarray,
    reference_spectra: np.ndarray,
    max_angle: Optional[float] = None,
    unclassified_val: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Spectral Angle Mapper (SAM) classification.

    Measures the spectral angle between each pixel vector and endmember reference spectra.
    Pixels with minimum spectral angle greater than max_angle are assigned unclassified_val.

    Args:
        cube: 3D numpy array of shape (bands, lines, samples).
        reference_spectra: 2D numpy array of shape (num_endmembers, bands).
        max_angle: Optional maximum spectral angle threshold in radians.
        unclassified_val: Value assigned to pixels exceeding max_angle or containing NaN (default -1).

    Returns:
        Tuple of (rule_images, classification_map)
        - rule_images: 3D array of shape (num_endmembers, lines, samples) with angles in radians.
        - classification_map: 2D array of shape (lines, samples) with integer class indices.
    """
    bands, lines, samples = cube.shape
    num_endmembers = reference_spectra.shape[0]

    # Reshape cube to (bands, N)
    pixel_vectors = cube.reshape(bands, -1).astype(np.float32)  # shape (bands, N)

    # Detect invalid pixels (NaN or Inf)
    invalid_mask = np.any(~np.isfinite(pixel_vectors), axis=0)

    # Calculate L2 norm of pixel vectors
    pixel_norms = np.linalg.norm(pixel_vectors, axis=0, keepdims=True)  # (1, N)
    pixel_norms = np.maximum(pixel_norms, 1e-8)

    # Normalize pixel vectors
    normed_pixels = pixel_vectors / pixel_norms  # (bands, N)

    # Calculate L2 norm of reference spectra
    ref_norms = np.linalg.norm(reference_spectra, axis=1, keepdims=True)  # (M, 1)
    ref_norms = np.maximum(ref_norms, 1e-8)
    normed_refs = reference_spectra / ref_norms  # (M, bands)

    # Cosine similarity: (M, bands) @ (bands, N) -> (M, N)
    cos_sim = np.dot(normed_refs, normed_pixels)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # Spectral angle in radians
    angles = np.arccos(cos_sim)  # (M, N)

    # Minimum angle determines assigned class
    class_indices = np.argmin(angles, axis=0).astype(np.int32)

    if max_angle is not None:
        min_angles = np.min(angles, axis=0)
        class_indices[min_angles > float(max_angle)] = unclassified_val

    if np.any(invalid_mask):
        class_indices[invalid_mask] = unclassified_val
        angles[:, invalid_mask] = np.nan

    class_map = class_indices.reshape(lines, samples).astype(np.int32)
    rule_images = angles.reshape(num_endmembers, lines, samples).astype(np.float32)

    return rule_images, class_map


def spectral_information_divergence(
    cube: np.ndarray,
    reference_spectra: np.ndarray,
    max_divergence: Optional[float] = None,
    unclassified_val: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Spectral Information Divergence (SID) classification.

    Computes the symmetric relative entropy (Kullback-Leibler divergence) between
    probability distributions derived from pixel spectra and reference endmember spectra:
        SID(p, q) = D(p || q) + D(q || p) = sum((p_i - q_i) * (ln(p_i) - ln(q_i)))

    Args:
        cube: 3D numpy array of shape (bands, lines, samples).
        reference_spectra: 2D numpy array of shape (num_endmembers, bands).
        max_divergence: Optional maximum divergence threshold. Pixels with divergence
            greater than this value are set to unclassified_val.
        unclassified_val: Value assigned to unclassified/invalid pixels (default -1).

    Returns:
        Tuple of (rule_images, classification_map):
        - rule_images: 3D array of shape (num_endmembers, lines, samples) containing SID values.
        - classification_map: 2D array of shape (lines, samples) with integer class indices.
    """
    bands, lines, samples = cube.shape
    num_endmembers = reference_spectra.shape[0]

    pixel_vectors = cube.reshape(bands, -1).astype(np.float64)  # (bands, N)
    num_pixels = pixel_vectors.shape[1]

    # Detect invalid (NaN, Inf, or all-zero / negative sum) pixels
    invalid_mask = np.any(~np.isfinite(pixel_vectors), axis=0)

    # Normalize reference spectra to probability vectors: sum(q) = 1
    ref_pos = np.maximum(reference_spectra.astype(np.float64), 1e-8)
    q = ref_pos / np.sum(ref_pos, axis=1, keepdims=True)  # (M, bands)
    log_q = np.log(q)  # (M, bands)

    # Normalize pixel spectra to probability vectors: sum(p) = 1
    p_pos = np.maximum(pixel_vectors, 1e-8)
    p_sum = np.sum(p_pos, axis=0, keepdims=True)  # (1, N)
    p = p_pos / np.maximum(p_sum, 1e-8)  # (bands, N)
    log_p = np.log(p)  # (bands, N)

    # Compute symmetric divergence:
    # SID(p, q) = sum_b (p_b - q_b) * (log(p_b) - log(q_b))
    #           = sum_b [ p_b * log(p_b) - p_b * log(q_b) - q_b * log(p_b) + q_b * log(q_b) ]
    #
    # Term 1: p * log(p) sum over bands -> (1, N)
    term1 = np.sum(p * log_p, axis=0, keepdims=True)  # (1, N)
    # Term 2: q @ log(p) -> (M, N)
    term2 = np.dot(q, log_p)  # (M, N)
    # Term 3: log(q) @ p -> (M, N)
    term3 = np.dot(log_q, p)  # (M, N)
    # Term 4: q * log(q) sum over bands -> (M, 1)
    term4 = np.sum(q * log_q, axis=1, keepdims=True)  # (M, 1)

    # sid = term1 - term2 - term3 + term4  # (M, N)
    sid = (term1 + term4) - (term2 + term3)
    sid = np.maximum(sid, 0.0)  # clamp non-negative

    class_indices = np.argmin(sid, axis=0).astype(np.int32)

    if max_divergence is not None:
        min_sid = np.min(sid, axis=0)
        class_indices[min_sid > float(max_divergence)] = unclassified_val

    if np.any(invalid_mask):
        class_indices[invalid_mask] = unclassified_val
        sid[:, invalid_mask] = np.nan

    class_map = class_indices.reshape(lines, samples).astype(np.int32)
    rule_images = sid.reshape(num_endmembers, lines, samples).astype(np.float32)

    return rule_images, class_map


def compute_ica(
    cube: np.ndarray,
    num_components: int = 3,
    max_iter: int = 200,
    tol: float = 1e-4,
    algorithm: str = "parallel",
    fun: str = "logcosh",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Independent Component Analysis (ICA) using FastICA.

    Separates mixed hyperspectral/multispectral signals into statistically
    independent components for spectral unmixing and anomaly detection.

    Args:
        cube: 3D numpy array of shape (bands, lines, samples).
        num_components: Number of independent components to extract.
        max_iter: Maximum number of FastICA iterations.
        tol: Convergence tolerance.
        algorithm: FastICA algorithm ('parallel' or 'deflation').
        fun: G-function form ('logcosh', 'exp', or 'cube').
        random_state: Random state seed.

    Returns:
        Tuple of (ica_cube, mixing_matrix):
        - ica_cube: np.ndarray of shape (num_components, lines, samples)
        - mixing_matrix: estimated mixing matrix of shape (bands, num_components)
    """
    from sklearn.decomposition import FastICA

    bands, lines, samples = cube.shape
    num_components = min(num_components, bands)
    num_pixels = lines * samples

    flat_data = cube.reshape(bands, num_pixels).T.astype(np.float32)
    finite_pixels = np.all(np.isfinite(flat_data), axis=1)

    # FastICA rejects non-finite values, so fit and transform on the clean pixels
    # and scatter any non-finite pixels back as NaN in the result.
    clean_data = flat_data[finite_pixels]
    if len(clean_data) < num_components + 1:
        raise ValueError(
            f"ICA requires at least {num_components + 1} valid pixels, found {len(clean_data)}."
        )

    # Subsample for fast convergence on large images (>250k pixels)
    if len(clean_data) > 250_000:
        step = max(1, int(np.sqrt(len(clean_data) / 100_000)))
        sub_data = clean_data[::step]
    else:
        sub_data = clean_data

    ica = FastICA(
        n_components=num_components,
        algorithm=algorithm,
        fun=fun,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        whiten="unit-variance",
    )
    ica.fit(sub_data)

    # Transform full raster in memory-safe chunks
    transformed = np.full((num_pixels, num_components), np.nan, dtype=np.float32)
    chunk_size = 100_000
    for start in range(0, len(clean_data), chunk_size):
        end = min(start + chunk_size, len(clean_data))
        chunk = clean_data[start:end]
        transformed[np.where(finite_pixels)[0][start:end]] = ica.transform(chunk)

    ica_cube = transformed.T.reshape(num_components, lines, samples).astype(np.float32)
    mixing_matrix = ica.mixing_ if hasattr(ica, "mixing_") else np.eye(bands, num_components)

    return ica_cube, mixing_matrix

