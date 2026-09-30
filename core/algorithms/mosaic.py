"""OpenENVI Mosaicking Engine.

Combines georeferenced rasters onto one common grid. Unlike stacking, which only
rebinds bands, a mosaic has to agree on where pixels actually sit on the ground:
bounds are reprojected into a common CRS, a union grid is built at the reference
resolution, and every source is resampled onto it. Overlapping regions are blended
with an optional linear feather so seams do not show as hard edges.
"""

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

import core.proj_setup  # Ensure PROJ library configuration is loaded
import rasterio.warp
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterMetadata

def _as_affine(transform) -> Affine:
    """Coerce whatever the readers store (Affine or tuple) into an Affine."""
    if transform is None:
        raise ValueError("Raster has no geotransform; mosaicking needs georeferenced input.")
    if isinstance(transform, Affine):
        return transform
    return Affine(*tuple(transform)[:6])


def raster_bounds(reader: BaseRasterReader) -> Tuple[float, float, float, float]:
    """Return the (left, bottom, right, top) footprint of a reader in its own CRS."""
    meta = reader.metadata
    if meta.transform is None:
        raise ValueError("Raster has no geotransform; mosaicking needs georeferenced input.")
    aff = _as_affine(meta.transform)
    # Affine maps (col, row) -> (x, y); row 0 is the top edge.
    left, top = aff * (0, 0)
    right, bottom = aff * (meta.width, meta.height)
    return min(left, right), min(bottom, top), max(left, right), max(bottom, top)


def build_mosaic_grid(
    readers: Sequence[BaseRasterReader],
    target_crs: Optional[str] = None,
    resolution: Optional[Tuple[float, float]] = None,
) -> Tuple[Affine, int, int, str]:
    """Compute the union grid that covers every input.

    Args:
        readers: Rasters to combine. The first one supplies the reference resolution
            unless ``resolution`` overrides it.
        target_crs: CRS to mosaic into. Defaults to the first raster's CRS.
        resolution: Explicit (x_res, y_res) in target CRS units. Defaults to the
            first raster's pixel size, which keeps the output predictable and never
            silently upsamples a coarse source.

    Returns:
        Tuple of (transform, width, height, crs)
    """
    if not readers:
        raise ValueError("No input rasters provided for mosaicking.")

    crs_list = [r.metadata.crs for r in readers]
    if any(c is None for c in crs_list):
        raise ValueError("Every input raster must carry a CRS to be mosaicked.")

    dst_crs = target_crs or crs_list[0]

    # Union of the footprints once they are all expressed in the target CRS.
    union = None
    for reader, src_crs in zip(readers, crs_list):
        if str(src_crs) == str(dst_crs):
            bounds = raster_bounds(reader)
        else:
            bounds = transform_bounds(src_crs, dst_crs, *raster_bounds(reader))
        if union is None:
            union = list(bounds)
        else:
            union[0] = min(union[0], bounds[0])
            union[1] = min(union[1], bounds[1])
            union[2] = max(union[2], bounds[2])
            union[3] = max(union[3], bounds[3])
    minx, miny, maxx, maxy = union

    if resolution is not None:
        x_res, y_res = float(resolution[0]), float(resolution[1])
    else:
        ref = _as_affine(readers[0].metadata.transform)
        if str(crs_list[0]) != str(dst_crs):
            # The reference pixel size only stays valid if its CRS is unchanged.
            raise ValueError(
                "Reference raster CRS differs from the target CRS; pass an explicit "
                "resolution to mosaic it."
            )
        x_res, y_res = abs(ref.a), abs(ref.e)

    if x_res <= 0 or y_res <= 0:
        raise ValueError("Mosaic resolution must be positive.")

    width = max(1, int(math.ceil((maxx - minx) / x_res)))
    height = max(1, int(math.ceil((maxy - miny) / y_res)))
    # North-up: the y step must be negative, otherwise row 0 walks north from the
    # top edge and the grid lands above the data instead of over it.
    transform = Affine.translation(minx, maxy) * Affine.scale(x_res, -y_res)
    return transform, width, height, dst_crs


def _destination_window(
    reader: BaseRasterReader,
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    dst_crs: str,
) -> Tuple[int, int, int, int, Affine]:
    """Locate a source inside the output grid.

    Returns the (row_off, col_off, height, width) slice it covers and the transform
    describing that slice, so the source only gets resampled over the region it
    actually occupies instead of the whole mosaic.
    """
    if str(reader.metadata.crs) == str(dst_crs):
        left, bottom, right, top = raster_bounds(reader)
    else:
        left, bottom, right, top = transform_bounds(
            reader.metadata.crs, dst_crs, *raster_bounds(reader)
        )

    inverse = ~dst_transform
    col_start, row_start = inverse * (left, top)
    col_end, row_end = inverse * (right, bottom)

    col_off = max(0, int(math.floor(col_start)))
    row_off = max(0, int(math.floor(row_start)))
    col_end = min(dst_width, int(math.ceil(col_end)))
    row_end = min(dst_height, int(math.ceil(row_end)))

    width = max(0, col_end - col_off)
    height = max(0, row_end - row_off)
    sub_transform = dst_transform * Affine.translation(col_off, row_off)
    return row_off, col_off, height, width, sub_transform


def _feather_weight(height: int, width: int, feather_pixels: float) -> np.ndarray:
    """Separable ramp that fades a source in over ``feather_pixels`` at its edges.

    Without feathering a source contributes 1 everywhere, which is the right
    behaviour when inputs do not overlap and harmless when they do not touch.
    """
    if feather_pixels <= 0 or height <= 0 or width <= 0:
        return np.ones((height, width), dtype=np.float32)

    def ramp(n: int) -> np.ndarray:
        # Distance in pixels from each edge, then rescaled so it reaches 1.
        centre = (np.arange(n) + 0.5)
        edge = np.minimum(centre, n - centre)
        return np.clip(edge / feather_pixels, 0.0, 1.0).astype(np.float32)

    return np.outer(ramp(height), ramp(width))


def _resampling(name: str):
    """Resolve a resampling name, tolerating the GDAL spelling of cubic spline."""
    key = name.lower().replace(" ", "")
    aliases = {"cubicspline": "cubic_spline", "cubics": "cubic"}
    key = aliases.get(key, key)
    try:
        return rasterio.warp.Resampling[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown resampling method {name!r}; expected one of "
            "nearest, bilinear, cubic, cubic_spline, average, lanczos."
        ) from exc


def mosaic_rasters(
    readers: Sequence[BaseRasterReader],
    target_crs: Optional[str] = None,
    resolution: Optional[Tuple[float, float]] = None,
    resample_method: str = "bilinear",
    background_value: float = 0.0,
    feather_pixels: float = 0.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, RasterMetadata]:
    """Mosaic georeferenced rasters onto one grid.

    Args:
        readers: Rasters to combine. All must share a band count.
        target_crs: CRS to mosaic into. Defaults to the first raster's CRS.
        resolution: Explicit (x_res, y_res). Defaults to the first raster's pixel size.
        resample_method: 'nearest', 'bilinear', 'cubic', 'cubicspline' or 'lanczos'.
        background_value: Value for pixels no input covers.
        feather_pixels: Blend width in pixels across each source's edge. Use this
            whenever inputs overlap, otherwise the seam is a hard discontinuity.
        progress_callback: Optional (current, total) progress callback.

    Returns:
        Tuple of (cube, metadata) where cube has shape (height, width, bands).

    Raises:
        ValueError: If no inputs are given, a CRS or geotransform is missing, or the
            inputs disagree on band count.
    """
    if not readers:
        raise ValueError("No input rasters provided for mosaicking.")

    band_counts = {r.metadata.bands for r in readers}
    if len(band_counts) != 1:
        raise ValueError(
            f"All inputs must have the same band count; got {sorted(band_counts)}."
        )
    total_bands = band_counts.pop()

    dst_transform, dst_width, dst_height, dst_crs = build_mosaic_grid(
        readers, target_crs=target_crs, resolution=resolution
    )
    resampling = _resampling(resample_method)

    cube = np.full((dst_height, dst_width, total_bands), background_value, dtype=np.float32)

    for band in range(total_bands):
        acc = np.zeros((dst_height, dst_width), dtype=np.float32)
        band_weights = np.zeros((dst_height, dst_width), dtype=np.float32)

        for reader in readers:
            row_off, col_off, height, width, sub_transform = _destination_window(
                reader, dst_transform, dst_width, dst_height, dst_crs
            )
            if height <= 0 or width <= 0:
                # This source falls entirely outside the union grid.
                continue

            source = reader.read_band(band)
            if source.shape != (reader.metadata.height, reader.metadata.width):
                source = np.asarray(source)[: reader.metadata.height, : reader.metadata.width]

            # NaN marks uncovered pixels, which keeps a legitimate reflectance of 0
            # distinguishable from "this source has nothing here".
            target = np.full((height, width), np.nan, dtype=np.float32)
            rasterio.warp.reproject(
                source=np.ascontiguousarray(source, dtype=np.float32),
                src_transform=_as_affine(reader.metadata.transform),
                src_crs=reader.metadata.crs,
                src_nodata=reader.metadata.nodata,
                destination=target,
                dst_transform=sub_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )

            valid = np.isfinite(target)
            if not valid.any():
                continue

            weight = _feather_weight(height, width, feather_pixels) * valid
            window_rows = slice(row_off, row_off + height)
            window_cols = slice(col_off, col_off + width)
            acc[window_rows, window_cols] += weight * np.nan_to_num(target)
            band_weights[window_rows, window_cols] += weight

        covered = band_weights > 0
        # Divide only where something actually contributed, so background_value in
        # genuinely empty areas survives instead of becoming NaN.
        np.divide(acc, band_weights, out=cube[:, :, band], where=covered)
        if not covered.all():
            cube[~covered, band] = background_value

        if progress_callback is not None:
            progress_callback(band + 1, total_bands)

    ref_meta = readers[0].metadata
    band_details: List[BandInfo] = list(ref_meta.band_details or [])
    if len(band_details) != total_bands:
        band_details = [
            BandInfo(index=i, name=f"Band {i + 1}", wavelength=None, mtl_band=i + 1)
            for i in range(total_bands)
        ]

    # A mosaic edge is an area no input imaged, so it is genuinely no-data.
    meta = RasterMetadata(
        width=dst_width,
        height=dst_height,
        bands=total_bands,
        dtype="float32",
        crs=dst_crs,
        transform=tuple(dst_transform)[:6],
        interleave="BSQ",
        nodata=background_value,
        band_details=band_details,
    )
    return cube, meta
