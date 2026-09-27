"""OpenENVI Core Module.

Contains event bus, data models, and core remote sensing abstractions.
"""

import core.proj_setup  # Must run before rasterio is imported
from .events import AppEventBus, event_bus
from .i18n import i18n, tr
from .models import BandInfo, RasterLayer, RasterMetadata

__all__ = [
    "AppEventBus",
    "event_bus",
    "i18n",
    "tr",
    "BandInfo",
    "RasterMetadata",
    "RasterLayer",
]
