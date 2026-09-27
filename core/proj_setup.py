"""OpenENVI PROJ Environment Configuration Helper.

Ensures Rasterio / GDAL uses its bundled PROJ database to prevent conflicts
with external software (such as PostgreSQL / PostGIS) on Windows.
"""

import importlib.util
import os


def configure_proj_env() -> None:
    """Detect and enforce Rasterio's bundled PROJ database."""
    try:
        spec = importlib.util.find_spec("rasterio")
        if spec and spec.submodule_search_locations:
            r_dir = spec.submodule_search_locations[0]
            proj_data = os.path.join(r_dir, "proj_data")
            if os.path.isdir(proj_data):
                os.environ["PROJ_LIB"] = proj_data
                os.environ["PROJ_DATA"] = proj_data
    except Exception:
        pass


configure_proj_env()
