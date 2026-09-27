"""Pytest configuration and global setup for OpenENVI tests."""

import os

# Set headless platform before any Qt widgets are loaded
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Configure temporary directory to drive with available space
import tempfile
os.makedirs(r"E:\temp", exist_ok=True)
os.environ["TEMP"] = r"E:\temp"
os.environ["TMP"] = r"E:\temp"
tempfile.tempdir = r"E:\temp"

# Configure PROJ database paths before rasterio is loaded
import core.proj_setup
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Reset QSettings to clean state for test predictability
from PySide6.QtCore import QSettings
from core.i18n import i18n
QSettings("OpenENVI", "OpenENVI").clear()
i18n.set_language("en")
