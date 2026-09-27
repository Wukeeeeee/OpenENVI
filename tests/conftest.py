"""Pytest configuration and global setup for OpenENVI tests."""

import os

# Set headless platform before any Qt widgets are loaded
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Configure PROJ database paths before rasterio is loaded
import core.proj_setup
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
