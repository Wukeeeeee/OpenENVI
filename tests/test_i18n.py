"""Tests for OpenENVI Internationalization (i18n) and Language Switching.

Verifies dynamic translation between English and Chinese across UI components.
"""

import os
import pytest
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from app.main_window import OpenENVIMainWindow
from core.i18n import i18n, tr


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_i18n_translations():
    """Verify dictionary translation lookups."""
    i18n.set_language("en")
    assert tr("app.title") == "OpenENVI - Remote Sensing Platform"
    assert tr("dock.layer_manager") == "Layer Manager"
    assert tr("dock.data_manager") == "Data Manager"

    i18n.set_language("zh")
    assert tr("app.title") == "OpenENVI - 遥感与高光谱分析平台"
    assert tr("dock.layer_manager") == "图层管理器"
    assert tr("dock.data_manager") == "数据管理器 (可用波段)"

    # Reset to default
    i18n.set_language("en")


def test_main_window_language_switch(qapp):
    """Verify that switching language updates all dock titles and menus in real time."""
    window = OpenENVIMainWindow()

    # Initial state (English)
    i18n.set_language("en")
    assert "OpenENVI - Remote Sensing Platform" in window.windowTitle()
    assert window.dock_layer_manager.windowTitle() == "Layer Manager"
    assert window.dock_toolbox.windowTitle() == "Toolbox"
    assert window.dock_spectral_profile.windowTitle() == "Spectral Profile (Z-Profile)"

    # Switch to Chinese
    i18n.set_language("zh")
    assert "OpenENVI - 遥感与高光谱分析平台" in window.windowTitle()
    assert window.dock_layer_manager.windowTitle() == "图层管理器"
    assert window.dock_toolbox.windowTitle() == "ENVI 工具箱"
    assert window.dock_spectral_profile.windowTitle() == "波谱剖面 (Z-Profile)"
    assert window.menu_file.title() == "文件(&F)"
    assert window.menu_language.title() == "语言(&G)"

    # Verify toolbox child items translated to Chinese
    first_cat = window.dock_toolbox.tree.topLevelItem(0)
    assert "基础工具" in first_cat.text(0)
    assert first_cat.child(0).text(0) == "波段运算 (Band Math)"

    # Switch back to English
    i18n.set_language("en")
    assert "OpenENVI - Remote Sensing Platform" in window.windowTitle()
    assert window.dock_layer_manager.windowTitle() == "Layer Manager"
    assert window.dock_toolbox.windowTitle() == "Toolbox"
    assert window.menu_file.title() == "&File"
    assert "Basic Tools" in window.dock_toolbox.tree.topLevelItem(0).text(0)
    assert window.dock_toolbox.tree.topLevelItem(0).child(0).text(0) == "Band Math"

    window.close()
