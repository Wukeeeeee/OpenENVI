"""OpenENVI Professional Dark-Slate Stylesheet.

Engineered for high-density scientific desktop workflows, featuring sharp geometry,
crisp contrast, and a low visual-fatigue dark slate palette.
"""

DARK_THEME_QSS = """
/* Global Base Settings */
* {
    font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", "Inter", "Arial", sans-serif;
    font-size: 12px;
    color: #e1e4e8;
}

QMainWindow, QDialog {
    background-color: #1e2124;
}

/* ToolBar & ToolButtons */
QToolBar {
    background-color: #282b30;
    border-bottom: 1px solid #36393e;
    spacing: 4px;
    padding: 3px 6px;
}

QToolBar::separator {
    width: 1px;
    background-color: #424549;
    margin: 4px 6px;
}

QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 2px;
    padding: 3px 8px;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #36393e;
    border: 1px solid #4e535a;
}

QToolButton:pressed, QToolButton:checked {
    background-color: #32353b;
    border: 1px solid #7289da;
}

/* Menu Bar & Menus */
QMenuBar {
    background-color: #1e2124;
    border-bottom: 1px solid #2f3136;
    padding: 1px 4px;
}

QMenuBar::item {
    background-color: transparent;
    padding: 4px 10px;
    border-radius: 2px;
}

QMenuBar::item:selected {
    background-color: #2f3136;
}

QMenu {
    background-color: #282b30;
    border: 1px solid #424549;
    padding: 4px 0px;
}

QMenu::item {
    padding: 5px 24px 5px 20px;
}

QMenu::item:selected {
    background-color: #3a71c1;
    color: #ffffff;
}

QMenu::separator {
    height: 1px;
    background-color: #424549;
    margin: 4px 0px;
}

/* Dock Widgets */
QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    font-weight: 600;
}

QDockWidget::title {
    background-color: #282b30;
    border-bottom: 1px solid #36393e;
    text-align: left;
    padding: 5px 8px;
    color: #c9d1d9;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Status Bar */
QStatusBar {
    background-color: #1a1c1e;
    border-top: 1px solid #2f3136;
    color: #8e9297;
    font-size: 11px;
    padding: 2px 4px;
}

QStatusBar::item {
    border: none;
    padding: 0px 4px;
}

/* Tree & List Widgets */
QTreeWidget, QListWidget, QTableView {
    background-color: #23272a;
    border: 1px solid #2f3136;
    selection-background-color: #3a71c1;
    selection-color: #ffffff;
    alternate-background-color: #202225;
    outline: none;
}

QTreeWidget::item, QListWidget::item {
    padding: 4px 6px;
    border-radius: 2px;
}

QTreeWidget::item:hover, QListWidget::item:hover {
    background-color: #2f3136;
}

QTreeWidget::item:selected, QListWidget::item:selected {
    background-color: #3a71c1;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #282b30;
    color: #8e9297;
    padding: 4px 8px;
    border: none;
    border-right: 1px solid #36393e;
    border-bottom: 1px solid #36393e;
    font-size: 11px;
    font-weight: 600;
}

/* Buttons */
QPushButton {
    background-color: #2f3136;
    border: 1px solid #424549;
    border-radius: 2px;
    padding: 5px 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #36393e;
    border-color: #7289da;
}

QPushButton:pressed {
    background-color: #202225;
}

QPushButton:disabled {
    background-color: #1e2124;
    color: #5d6269;
    border-color: #2a2c30;
}

/* Combo Box */
QComboBox {
    background-color: #2f3136;
    border: 1px solid #424549;
    border-radius: 2px;
    padding: 3px 8px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #7289da;
}

QComboBox::drop-down {
    border: none;
    width: 18px;
}

QComboBox QAbstractItemView {
    background-color: #282b30;
    border: 1px solid #424549;
    selection-background-color: #3a71c1;
    selection-color: #ffffff;
}

/* Splitter & ScrollBars */
QSplitter::handle {
    background-color: #1e2124;
}

QScrollBar:vertical {
    background: #1e2124;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #36393e;
    min-height: 20px;
    border-radius: 2px;
}

QScrollBar::handle:vertical:hover {
    background: #4e535a;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}

QScrollBar:horizontal {
    background: #1e2124;
    height: 10px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #36393e;
    min-width: 20px;
    border-radius: 2px;
}

QScrollBar::handle:horizontal:hover {
    background: #4e535a;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
    width: 0px;
}

/* GroupBox & Labels */
QGroupBox {
    border: 1px solid #36393e;
    border-radius: 2px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: #8e9297;
}

QLabel {
    color: #c9d1d9;
}
"""
