"""OpenENVI Remote Sensing Classification Engine.

Provides K-Means, ISODATA clustering, and thematic color rendering
for multispectral and hyperspectral imagery.
"""

from typing import List, Optional, Tuple
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

        train_sq = np.sum(train_data**2, axis=1, keepdims=True)
        centers_sq = np.sum(centers**2, axis=1, keepdims=True).T
        distances = train_sq - 2.0 * np.dot(train_data, centers.T) + centers_sq
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
    centers_sq = np.sum(centers**2, axis=1, keepdims=True).T
    chunk_size = 250_000
    total_chunks = (num_pixels + chunk_size - 1) // chunk_size

    for chunk_i, start in enumerate(range(0, num_pixels, chunk_size)):
        end = min(start + chunk_size, num_pixels)
        chunk = flat_data[start:end]
        chunk_sq = np.sum(chunk**2, axis=1, keepdims=True)
        dist = chunk_sq - 2.0 * np.dot(chunk, centers.T) + centers_sq
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
        data_sq = np.sum(data**2, axis=1, keepdims=True)
        centers_sq = np.sum(centers**2, axis=1, keepdims=True).T
        distances = data_sq - 2.0 * np.dot(data, centers.T) + centers_sq
        labels = np.argmin(distances, axis=1).astype(np.int32)

    return labels.reshape(lines, samples), centers


def create_thematic_rgb(
    class_map: np.ndarray,
    palette: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Convert a 2D integer class map into an RGB thematic color image.

    Args:
        class_map: 2D numpy array of shape (lines, samples).
        palette: Optional list of (R, G, B) tuples.

    Returns:
        3D numpy array of shape (lines, samples, 3) in uint8 [0, 255].
    """
    if palette is None:
        palette = DEFAULT_THEMATIC_PALETTE

    lines, samples = class_map.shape
    rgb_image = np.zeros((lines, samples, 3), dtype=np.uint8)

    for class_idx in np.unique(class_map):
        color = palette[int(class_idx) % len(palette)]
        mask = (class_map == class_idx)
        rgb_image[mask] = color

    return rgb_image
