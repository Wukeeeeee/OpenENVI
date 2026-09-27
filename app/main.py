"""OpenENVI Application Entry Point.

Initializes the Qt application instance, applies the professional dark-slate theme,
instantiates the main window, and launches the desktop event loop.
"""

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.main_window import OpenENVIMainWindow
from ui.styles import DARK_THEME_QSS


def create_app() -> tuple[QApplication, OpenENVIMainWindow]:
    """Create and configure the QApplication and OpenENVIMainWindow instances."""
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("OpenENVI")
    app.setApplicationDisplayName("OpenENVI - Remote Sensing Platform")
    app.setOrganizationName("OpenENVI")
    app.setStyleSheet(DARK_THEME_QSS)

    window = OpenENVIMainWindow()
    return app, window


def main() -> int:
    """Main execution function."""
    import os
    app, window = create_app()

    # If --test-init is provided in sys.argv, exit immediately after clean initialization
    if "--test-init" in sys.argv:
        print("OpenENVI initialized successfully in test mode.")
        return 0

    # Auto-load file if passed via command line
    for arg in sys.argv[1:]:
        if not arg.startswith("--") and os.path.exists(arg):
            window.open_raster_file(arg)
            break

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
