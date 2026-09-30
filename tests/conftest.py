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

# Preload scikit-learn (and the pandas/pyarrow native stack it may pull in) before
# any Qt window exists. Importing these lazily from inside a live Qt application
# triggers a native access violation while loading pyarrow's extension module.
import sklearn.svm  # noqa: F401
import sklearn.decomposition  # noqa: F401

# Reset QSettings to clean state for test predictability
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from core.i18n import i18n
import pytest

QSettings("OpenENVI", "OpenENVI").clear()
i18n.set_language("en")


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single persistent QApplication instance for test suite."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

