"""OpenENVI Hyperspectral Algorithms Engine.

Provides Principal Component Analysis (PCA), Minimum Noise Fraction (MNF),
and Spectral Angle Mapper (SAM) for hyperspectral data reduction and classification.
"""

from typing import Tuple
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

    # Reshape (bands, lines, samples) to (N, bands) where N = lines * samples
    flat_data = cube.reshape(bands, -1).T  # shape (N, bands)
    mean_vec = np.mean(flat_data, axis=0, keepdims=True)
    centered = flat_data - mean_vec

    # Compute covariance matrix (bands, bands)
    cov = np.cov(centered, rowvar=False)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = np.sum(eigenvalues)
    explained_var = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)

    # Project centered data onto top eigenvectors
    top_eigenvectors = eigenvectors[:, :num_components]
    scores = np.dot(centered, top_eigenvectors)  # (N, num_components)

    # Reshape back to (num_components, lines, samples)
    score_cube = scores.T.reshape(num_components, lines, samples).astype(np.float32)

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

    # 1. Estimate noise from spatial shift differences: Delta = X[:, :, 1:] - X[:, :, :-1]
    diff_h = cube[:, :, 1:] - cube[:, :, :-1]
    diff_flat = diff_h.reshape(bands, -1).T  # shape (N_diff, bands)
    noise_cov = np.cov(diff_flat, rowvar=False) * 0.5  # variance of difference is 2 * noise_var

    # Regularize noise covariance for numerical stability
    noise_cov += np.eye(bands) * 1e-6

    # 2. Total data covariance
    data_flat = cube.reshape(bands, -1).T
    data_centered = data_flat - np.mean(data_flat, axis=0, keepdims=True)
    total_cov = np.cov(data_centered, rowvar=False)

    # 3. Solve generalized eigenvalue problem: total_cov * V = noise_cov * V * D
    # Equivalent to inv(noise_cov) * total_cov
    try:
        inv_noise = np.linalg.pinv(noise_cov)
        mat = np.dot(inv_noise, total_cov)
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        # Take real parts
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        top_vectors = eigenvectors[:, :num_components]
        mnf_scores = np.dot(data_centered, top_vectors)
        mnf_cube = mnf_scores.T.reshape(num_components, lines, samples).astype(np.float32)

        return mnf_cube, eigenvalues[:num_components]
    except Exception:
        # Fallback to standard PCA if generalized inversion encounters singularity
        score_cube, eig, _ = compute_pca(cube, num_components=num_components)
        return score_cube, eig[:num_components]


def spectral_angle_mapper(
    cube: np.ndarray,
    reference_spectra: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Spectral Angle Mapper (SAM) classification.

    Measures the spectral angle between each pixel vector and endmember reference spectra.

    Args:
        cube: 3D numpy array of shape (bands, lines, samples).
        reference_spectra: 2D numpy array of shape (num_endmembers, bands).

    Returns:
        Tuple of (rule_images, classification_map)
        - rule_images: 3D array of shape (num_endmembers, lines, samples) with angles in radians.
        - classification_map: 2D array of shape (lines, samples) with integer class indices.
    """
    bands, lines, samples = cube.shape
    num_endmembers = reference_spectra.shape[0]

    # Reshape cube to (bands, N)
    pixel_vectors = cube.reshape(bands, -1).astype(np.float32)  # shape (bands, N)

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
    class_map = np.argmin(angles, axis=0).reshape(lines, samples).astype(np.int32)
    rule_images = angles.reshape(num_endmembers, lines, samples).astype(np.float32)

    return rule_images, class_map
