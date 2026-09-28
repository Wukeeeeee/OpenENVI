import os
from typing import Callable, List, Optional, Tuple
import numpy as np

import core.proj_setup  # Ensure PROJ library configuration is loaded
from rasterio.transform import Affine
from core.io.base import BaseRasterReader
from core.models import BandInfo, RasterMetadata


def stack_bands(
    band_sources: List[Tuple[BaseRasterReader, int]],
    resampling_method: str = "nearest",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, RasterMetadata]:
    """Stack specified bands from one or more readers into a single multi-band array.

    Args:
        band_sources: List of (reader, band_index) tuples in desired stack order.
        resampling_method: Resampling method ('nearest', 'bilinear', 'bicubic').
        progress_callback: Optional (current, total) progress callback.

    Returns:
        Tuple of:
            - 3D numpy array of shape (height, width, len(band_sources))
            - Combined RasterMetadata with synthesized band details
    """
    if not band_sources:
        raise ValueError("No band sources provided for stacking.")

    # Reference grid is defined by the first band
    ref_reader, ref_band_idx = band_sources[0]
    ref_meta = ref_reader.metadata
    target_h = ref_meta.height
    target_w = ref_meta.width

    total_bands = len(band_sources)
    stacked_cube = np.empty((target_h, target_w, total_bands), dtype=np.float32)
    combined_band_details: List[BandInfo] = []

    order_map = {"nearest": 0, "bilinear": 1, "bicubic": 3}
    resample_order = order_map.get(resampling_method.lower(), 0)

    for i, (reader, band_idx) in enumerate(band_sources):
        data = reader.read_band(band_idx)

        # Reproject or resample if coordinates or dimensions differ
        needs_alignment = (data.shape != (target_h, target_w))
        if (
            ref_meta.crs
            and reader.metadata.crs
            and ref_meta.transform
            and reader.metadata.transform
            and (needs_alignment or str(reader.metadata.transform) != str(ref_meta.transform))
        ):
            try:
                import rasterio.warp
                from rasterio.crs import CRS

                src_crs = CRS.from_string(reader.metadata.crs)
                dst_crs = CRS.from_string(ref_meta.crs)
                src_trans = (
                    reader.metadata.transform
                    if isinstance(reader.metadata.transform, Affine)
                    else Affine(*reader.metadata.transform[:6])
                )
                dst_trans = (
                    ref_meta.transform
                    if isinstance(ref_meta.transform, Affine)
                    else Affine(*ref_meta.transform[:6])
                )

                warp_resamp = {
                    "nearest": rasterio.warp.Resampling.nearest,
                    "bilinear": rasterio.warp.Resampling.bilinear,
                    "bicubic": rasterio.warp.Resampling.cubic,
                }.get(resampling_method.lower(), rasterio.warp.Resampling.nearest)

                aligned = np.zeros((target_h, target_w), dtype=np.float32)
                rasterio.warp.reproject(
                    source=data.astype(np.float32),
                    destination=aligned,
                    src_transform=src_trans,
                    src_crs=src_crs,
                    dst_transform=dst_trans,
                    dst_crs=dst_crs,
                    resampling=warp_resamp,
                    dst_nodata=ref_meta.nodata if ref_meta.nodata is not None else 0.0,
                )
                data = aligned
            except Exception:
                if data.shape != (target_h, target_w):
                    from core.algorithms.pansharpen import resample_band_to_grid
                    data = resample_band_to_grid(data, target_h, target_w, order=resample_order)
        elif data.shape != (target_h, target_w):
            from core.algorithms.pansharpen import resample_band_to_grid
            data = resample_band_to_grid(data, target_h, target_w, order=resample_order)

        stacked_cube[:, :, i] = data

        # Extract band name, wavelength, fwhm
        r_name = (
            getattr(reader, "name", "")
            or getattr(reader, "_name", "")
            or os.path.splitext(os.path.basename(getattr(reader, "file_path", "Layer")))[0]
        )
        band_name = f"Band {i + 1}"
        wavelength = None
        wl_unit = "nm"
        fwhm = None

        if reader.metadata.band_details and band_idx < len(reader.metadata.band_details):
            binfo = reader.metadata.band_details[band_idx]
            band_name = f"{r_name}: {binfo.name}" if binfo.name else f"{r_name}: B{band_idx + 1}"
            wavelength = binfo.wavelength
            wl_unit = binfo.wavelength_unit or "nm"
            fwhm = binfo.fwhm
        else:
            band_name = f"{r_name}: B{band_idx + 1}"

        combined_band_details.append(
            BandInfo(index=i, name=band_name, wavelength=wavelength, wavelength_unit=wl_unit, fwhm=fwhm)
        )

        if progress_callback:
            progress_callback(i + 1, total_bands)

    # Preserve transform as Affine
    ref_transform = ref_meta.transform
    if ref_transform is not None and not isinstance(ref_transform, Affine):
        try:
            ref_transform = Affine(*ref_transform[:6])
        except Exception:
            pass

    # Synthesize raw header if present
    combined_header = dict(ref_meta.raw_header) if ref_meta.raw_header else {}
    if combined_header:
        combined_header["bands"] = total_bands
        combined_header["samples"] = target_w
        combined_header["lines"] = target_h
        combined_header["band names"] = [b.name for b in combined_band_details]
        if all(b.wavelength is not None for b in combined_band_details):
            combined_header["wavelength"] = [f"{b.wavelength:.4f}" for b in combined_band_details]

    combined_metadata = RasterMetadata(
        width=target_w,
        height=target_h,
        bands=total_bands,
        dtype="float32",
        crs=ref_meta.crs,
        transform=ref_transform,
        nodata=ref_meta.nodata,
        band_details=combined_band_details,
        raw_header=combined_header,
    )

    return stacked_cube, combined_metadata
