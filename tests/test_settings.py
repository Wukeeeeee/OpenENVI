"""Automated Tests for Settings and Workspace Persistence.

Verifies saving and restoring window geometry, dock layouts,
and active language preference via QSettings.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from app.main_window import OpenENVIMainWindow
from core.i18n import i18n


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_settings_persistence(qapp):
    """Verify that language, geometry, and layout are persisted."""
    # Write test settings
    settings = QSettings("OpenENVI", "OpenENVI")
    settings.setValue("language", "zh")

    # Instantiate window; it should pick up "zh"
    window = OpenENVIMainWindow()
    assert i18n.current_language == "zh"
    assert window.act_lang_zh.isChecked()

    # Change to "en" and trigger closeEvent
    window._set_language("en")
    assert settings.value("language") == "en"

    # Close window to trigger closeEvent persistence
    window.close()

    # Re-verify QSettings stored "en"
    assert settings.value("language") == "en"
    assert settings.value("geometry") is not None
    assert settings.value("windowState") is not None
