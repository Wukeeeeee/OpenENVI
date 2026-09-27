"""OpenENVI Core Algorithms Package.

Hosts image processing, spectral analysis, contrast enhancements, and classification routines.
"""

from .classification import (
    DEFAULT_THEMATIC_PALETTE,
    create_thematic_rgb,
    isodata_clustering,
    kmeans_clustering,
)
from .indices import (
    calculate_evi,
    calculate_nbr,
    calculate_ndvi,
    calculate_ndwi,
    calculate_savi,
    evaluate_band_math,
)
from .spectral import (
    compute_mnf,
    compute_pca,
    spectral_angle_mapper,
)
from .stretch import (
    apply_stretch,
    gaussian_stretch,
    histogram_equalization_stretch,
    linear_percent_stretch,
    min_max_stretch,
)

__all__ = [
    # Stretch
    "apply_stretch",
    "linear_percent_stretch",
    "histogram_equalization_stretch",
    "gaussian_stretch",
    "min_max_stretch",
    # Indices & Math
    "calculate_ndvi",
    "calculate_ndwi",
    "calculate_evi",
    "calculate_savi",
    "calculate_nbr",
    "evaluate_band_math",
    # Spectral
    "compute_pca",
    "compute_mnf",
    "spectral_angle_mapper",
    # Classification
    "kmeans_clustering",
    "isodata_clustering",
    "create_thematic_rgb",
    "DEFAULT_THEMATIC_PALETTE",
]
